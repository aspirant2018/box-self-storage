from dotenv import load_dotenv

from livekit import agents
from livekit.plugins import (
    openai,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from typing import Literal, Annotated, Optional

from livekit.agents import (
    AgentSession,
    Agent,
    RoomInputOptions,
    function_tool,
    RunContext,
    get_job_context,
    beta
    )
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo  # Python 3.9+    
import logging
from dataclasses import dataclass, field
import yaml
from pydantic import Field
from livekit.agents import JobContext, WorkerOptions, cli
from helpers import send_post
from livekit import rtc


load_dotenv(".env.local")
logger = logging.getLogger("self-storage-agent")
logger.setLevel(logging.INFO)

webhook_url = "https://sought-cicada-scarcely.ngrok-free.app/webhook-test/5da5c003-23a5-407b-958b-a532daf7108c"  



instructions = """
# Role:
You are an AI voice assistant for a self-storage company. Your job is to answer inbound calls, talk naturally with customers, and collect their details.
# Instructions:
Greet the customer warmly and professionally.
Ask how you can assist them with their storage needs.
<Wait for the customer’s response>
If the customer wants to book a storage unit, ask for the following details in a friendly, conversational manner:
Desired unit size (small, medium, large)
Preferred location (Downtown, Uptown, Riverside)
<Wait for the customer’s response>
Use the 'check_availability' tool to see if there are units that match their requirements.
If units are available, provide the customer with the details (size, location, price) in a clear and friendly manner.
<Wait for the customer’s response>
Collect their contact information (name, phone number, and email) and confirm the booking.
Thank the customer for choosing your service and end the call politely.

# Guidelines:
Maintain a professional yet friendly tone throughout the conversation.
Adapt your language to be natural and easy to understand.
Confirm each piece of information politely before proceeding to the next step.
Be fast-keep responses short and snappy.
Sound human-sprinkle in light vocal pauses like 'Mmh…', 'Let me see…', or 'Alright…' at natural moments-but not too often.
Keep everything upbeat and easy to follow. Never overwhelm the customer, don't ask multiple questions at the same time. 

# Constraints:
Do not give out information unrelated to storage.
If unsure, politely say: “I’ll connect you to a representative who can assist further.”
Avoid long robotic sentences; keep the conversation natural.

# Knowledge:
## locations:
We have locations in Downtown, Uptown, and Riverside.
## unit sizes:
### small:
5m2 $50/month, 6m2 $55/month, 7m2 $60/month, 8m2 $65/month, 9m2 $70/month.
### medium:
10m2 $90/month, 12m2 $100/month, 15m2 $120/month, 18m2 $140/month, 19m2 $150/month.
### large:
20m2 $150/month, 22m2 $170/month, 25m2 $200/month.
"""

@dataclass
class MySessionInfo:
    customer_name: str | None = None
    email: str | None = None
    phone_number: str | None = None

class Assistant(Agent):

    headers = {
                "accept": "application/json",
                "Content-Type": "application/json",
            }
    
    def __init__(self) -> None:
        super().__init__(instructions=instructions)



    @function_tool()
    async def check_availability(self,
                                 context: RunContext,
                                 location: Annotated[str, Field(description="The location to check availability for. Example values: Downtown, Uptown, Riverside")],
                                 size: Annotated[int, Field(description="The size of the unit to check availability for. Example values: 5m2, 10m2, 20m2")],
                                 #move_in_date: Annotated[str, Field(description="The desired move-in date in YYYY-MM-DD format. Example value: 2025-10-15")],
                                 ) -> str:
        """Called when the agent needs to check unit availability. based on location, size given by the user."""
        # Simulate checking availability

        data = {
            "location": location,
            "size":size,
        }

        response = await send_post(
            webhook_url=webhook_url,
            headers=self.headers,
            tool_name="check_availability",
            data=data
            )
        
        return {"message": response}

    @function_tool()
    async def book_unit(self,
                        context: RunContext,
                        name: Annotated[str, Field(description="The customer's full name.")],
                    ) -> str:
        """Called when the agent needs to book a unit for the customer."""
                    # Simulate booking the unit
        email_result = await beta.workflows.GetEmailTask(chat_ctx=self.chat_ctx)


        data = {
                "name" : name,
                "phone": context.session.userdata.phone_number,
                "email": email_result, 
                }
        response = await send_post(
                        webhook_url=webhook_url,
                        headers=self.headers,
                        tool_name="book_unit",
                        data=data
                        )
        
        return {"message": response}




async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    userdata = MySessionInfo()
    participant = await ctx.wait_for_participant(kind=[rtc.ParticipantKind.PARTICIPANT_KIND_SIP])
    logger.info(f"Participant {participant.identity} has joined the room.")
    logger.info(f"Participant sip phoneNumber: {participant.attributes['sip.phoneNumber']}")

    userdata.phone_number = participant.attributes.get('sip.phoneNumber')

    session = AgentSession[MySessionInfo](
        stt=openai.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
        userdata=userdata,
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` instead for best results
            noise_cancellation=noise_cancellation.BVCTelephony(), 
        ),
    )

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(
        entrypoint_fnc=entrypoint,
        agent_name="my-telephony-agent",
        initialize_process_timeout=60
        )
        )
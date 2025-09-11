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
from livekit import api


load_dotenv(".env.local")
logger = logging.getLogger("self-storage-agent")
logger.setLevel(logging.INFO)

webhook_url = "https://sought-cicada-scarcely.ngrok-free.app/webhook-test/5da5c003-23a5-407b-958b-a532daf7108c"  



instructions = """
# Role:
You are an AI voice assistant for a self-storage company. Your job is to answer inbound calls, talk naturally with customers, and collect their details.
# Instructions:
1. Greet the customer warmly and professionally.
2. Ask how you can assist them with their storage needs.
3. <Wait for the customer’s response>
4. If the customer wants to book a storage unit, ask for size and the location a friendly, conversational manner:
5. <Wait for the customer’s response>
6. Use the 'check_availability' tool to check if there are units that match their requirements.
7. If units are available, provide the customer with the details (size, location, price) in a clear and friendly manner.
8. <Wait for the customer’s response>
9. Collect their contact information (name and email) and confirm the booking.
10. Thank the customer for choosing your service and end the call politely.

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
We have locations in Epinay-sur-seine.
## unit sizes:
### Mini: 1m2 : 36 €,  2m2 : 67 €
### Petit:  3m2 : 93€,  4m2 : 121 €
### Moyen: 5m2 : 129 €, 6m2 : 147 €; 7m2 : 170 €; 8m2 : 190 €; 9m2 : 204 €; 10m2 : 219 €;
### Grand:  11m2 : 236 €; 12m2 : 251€; 13 m2 : 268 €; 14m2 : 286 €; 15m2 : 313 €;
### Maxi: 16m2 : 323 €;  17m2 : 351 €; 18m2 : 423 €; 20m2 : 501 €;

## Loyer TTC/mois hors assurance 
Assurance de 1 à 4 m2 : 9 €/mois
Assurance de 5 à 25 m2 : 14 €/mois
"""
from livekit.agents import AgentTask, function_tool

class CollectConsent(AgentTask[bool]):
    def __init__(self):
        super().__init__(
            instructions="Ask for recording consent and get a clear yes or no answer."
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(instructions="Ask for permission to record the call for quality assurance purposes.")

    @function_tool
    async def consent_given(self) -> None:
        """Use this when the user gives consent to record."""
        self.complete(True)

    @function_tool
    async def consent_denied(self) -> None:
        """Use this when the user denies consent to record."""
        self.complete(False)


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
                                 location: Annotated[str, Field(description="The location to check availability for.")],
                                 size: Annotated[int, Field(description="The size of the unit to check availability for")],
                                 #move_in_date: Annotated[str, Field(description="The desired move-in date in YYYY-MM-DD format. Example value: 2025-10-15")],
                                 ) -> str:
        """Called when the agent needs to check unit availability. based on location, size given by the user."""
        # Simulate checking availability

        data = {
            "location": location.lower(),
            "size":int(size),
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

        data = {
                "name" : name,
                "phone": context.session.userdata.phone_number,
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
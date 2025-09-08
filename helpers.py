import aiohttp
import asyncio


async def send_post(webhook_url:str, headers:dict, tool_name: str, data: dict) -> dict:
        """
        Helper function to send a POST request to the webhook URL.
        """
        async with aiohttp.ClientSession(headers=headers) as session:
            task = asyncio.create_task(
                session.post(
                    webhook_url,
                    json={
                        "tool": tool_name,
                        "data": data
                    },
                )
            )
            response = await task
            return await response.json()
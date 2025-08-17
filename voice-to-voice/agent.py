import logging
from dotenv import load_dotenv

from livekit.agents import (
    AgentSession,
    Agent,
    JobContext,
    RoomInputOptions,
    cli,
    WorkerOptions,
)
from livekit.plugins import (
    deepgram,
    openai,
    silero,
    # noise_cancellation,  # comment/remove if not available in your LiveKit version
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("voice-agent")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Today is January 1st, 2025. You are a voice assistant named Andrew, "
                "created by the wonderful Sinan Ozdemir. You speak with users via voice, "
                "using short and concise responses and avoiding unpronounceable punctuation."
            )
        )


async def entrypoint(ctx: JobContext):
    """Start the voice assistant session."""
    # Load VAD
    vad = silero.VAD.load()

    # Create the agent session
    session = AgentSession(
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),  # or google.LLM() if you prefer
        tts=openai.TTS(),
        vad=vad,
        turn_detection=MultilingualModel(),
    )

    # Start the session in the room
    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(),  # remove noise_cancellation if not available
    )

    # Instruct the agent to speak first
    await session.generate_reply(instructions="say hello to the user")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            # prewarm_fnc=None,  # optional: can define a prewarm function if needed
        )
    )

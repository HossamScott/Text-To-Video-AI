import edge_tts

async def generate_audio(text, output_filename, voice="en-AU-WilliamNeural"):
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(output_filename)



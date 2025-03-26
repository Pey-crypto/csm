import os
import torch
import torchaudio
import time
import cProfile
import pstats
import io
from huggingface_hub import hf_hub_download
from generator import load_csm_1b, Segment
from dataclasses import dataclass

# Disable Triton compilation
os.environ["NO_TORCH_COMPILE"] = "1"

# Default prompts are available at https://hf.co/sesame/csm-1b
prompt_filepath_conversational_a = hf_hub_download(
    repo_id="sesame/csm-1b",
    filename="prompts/conversational_a.wav"
)
prompt_filepath_conversational_b = hf_hub_download(
    repo_id="sesame/csm-1b",
    filename="prompts/conversational_b.wav"
)

SPEAKER_PROMPTS = {
    "conversational_a": {
        "text": (
            "like revising for an exam I'd have to try and like keep up the momentum because I'd "
            "start really early I'd be like okay I'm gonna start revising now and then like "
            "you're revising for ages and then I just like start losing steam I didn't do that "
            "for the exam we had recently to be fair that was a more of a last minute scenario "
            "but like yeah I'm trying to like yeah I noticed this yesterday that like Mondays I "
            "sort of start the day with this not like a panic but like a"
        ),
        "audio": prompt_filepath_conversational_a
    },
    "conversational_b": {
        "text": (
            "like a super Mario level. Like it's very like high detail. And like, once you get "
            "into the park, it just like, everything looks like a computer game and they have all "
            "these, like, you know, if, if there's like a, you know, like in a Mario game, they "
            "will have like a question block. And if you like, you know, punch it, a coin will "
            "come out. So like everyone, when they come into the park, they get like this little "
            "bracelet and then you can go punching question blocks around."
        ),
        "audio": prompt_filepath_conversational_b
    }
}

def load_prompt_audio(audio_path: str, target_sample_rate: int) -> torch.Tensor:
    """Load audio from file and resample it to the target sample rate"""
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = audio_tensor.squeeze(0)
    # Resample is lazy so we can always call it
    audio_tensor = torchaudio.functional.resample(
        audio_tensor, orig_freq=sample_rate, new_freq=target_sample_rate
    )
    return audio_tensor

def prepare_prompt(text: str, speaker: int, audio_path: str, sample_rate: int) -> Segment:
    """Prepare a prompt segment with text and audio"""
    audio_tensor = load_prompt_audio(audio_path, sample_rate)
    return Segment(text=text, speaker=speaker, audio=audio_tensor)

def save_profile_stats(profile, filename, title, lines=20000, console_output=True, summary_file=None):
    """Save profile statistics to a file and optionally print to console and summary file"""
    # Create profile directory if it doesn't exist
    os.makedirs("profile_results", exist_ok=True)
    
    # Save full stats to a file
    profile_path = os.path.join("profile_results", f"{filename}.prof")
    profile.dump_stats(profile_path)
    print(f"Full profile stats saved to: {profile_path}")
    
    # Get stats as string for both text file and potentially summary file
    s = io.StringIO()
    stats = pstats.Stats(profile, stream=s).sort_stats('cumulative')
    stats.print_stats(lines)
    stats_str = s.getvalue()
    
    # Save readable stats to a text file
    stats_path = os.path.join("profile_results", f"{filename}.txt")
    with open(stats_path, 'w') as f:
        f.write(f"--- {title} ---\n\n")
        f.write(stats_str)
    print(f"Readable profile stats saved to: {stats_path}")
    
    # Also print to console if requested
    if console_output:
        print(f"\n--- {title} ---")
        print(stats_str)
    
    # Write to summary file if provided
    if summary_file:
        summary_file.write(f"\n--- {title} ---\n\n")
        summary_file.write(stats_str)
        summary_file.write("\n")

def main():
    # Create timestamp for unique filenames
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Select the best available device, skipping MPS due to float64 limitations
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Create a summary file for timing results
    os.makedirs("profile_results", exist_ok=True)
    summary_path = os.path.join("profile_results", f"{timestamp}_summary.txt")
    
    with open(summary_path, 'w') as summary_file:
        summary_file.write(f"CSM-1B Audio Generation Profile - {timestamp}\n")
        summary_file.write(f"Device: {device}\n\n")
        
        # Profiling model loading
        print("Loading model...")
        model_profile = cProfile.Profile()
        model_profile.enable()
        
        start_time = time.time()
        generator = load_csm_1b(device)
        model_load_time = time.time() - start_time
        
        model_profile.disable()
        save_profile_stats(
            model_profile, 
            f"{timestamp}_model_loading", 
            "Model Loading Profile",
            summary_file=summary_file
        )
        
        summary_file.write(f"Model loading time: {model_load_time:.2f} seconds\n\n")

        print("Preparing conversation data...")
        # Prepare prompts
        prompt_a = prepare_prompt(
            SPEAKER_PROMPTS["conversational_a"]["text"],
            0,
            SPEAKER_PROMPTS["conversational_a"]["audio"],
            generator.sample_rate
        )

        prompt_b = prepare_prompt(
            SPEAKER_PROMPTS["conversational_b"]["text"],
            1,
            SPEAKER_PROMPTS["conversational_b"]["audio"],
            generator.sample_rate
        )

        # Generate conversation
        conversation = [
            {"text": "Hey how are you doing?", "speaker_id": 0},
            {"text": "Pretty good, pretty good. How about you?", "speaker_id": 1},
            {"text": "I'm great! So happy to be speaking with you today.", "speaker_id": 0},
            {"text": "Me too! This is some cool stuff, isn't it?", "speaker_id": 1}
        ]

        # Generate each utterance
        generated_segments = []
        prompt_segments = [prompt_a, prompt_b]
        
        # Track generation times and profiles
        generation_times = []
        total_audio_duration = 0

        for i, utterance in enumerate(conversation):
            utterance_text = utterance["text"]
            utterance_speaker = utterance["speaker_id"]
            
            print(f"\nGenerating [{i+1}/{len(conversation)}]: {utterance_text}")
            summary_file.write(f"Utterance {i+1}: \"{utterance_text}\"\n")
            summary_file.write(f"  Speaker: {utterance_speaker}\n")
            
            # Start timing
            start_time = time.time()
            
            # Profile the generation
            profile = cProfile.Profile()
            profile.enable()
            
            # Generate the audio using the generator's API
            audio_tensor = generator.generate(
                text=utterance_text,
                speaker=utterance_speaker,
                context=prompt_segments + generated_segments,
                max_audio_length_ms=10_000,
            )
            
            profile.disable()
            
            # Calculate generation time
            generation_time = time.time() - start_time
            
            # Calculate audio duration in seconds
            audio_duration = audio_tensor.shape[0] / generator.sample_rate
            total_audio_duration += audio_duration
            
            # Save timing metrics
            generation_times.append({
                "utterance": utterance_text,
                "speaker": utterance_speaker,
                "generation_time_seconds": generation_time,
                "audio_duration_seconds": audio_duration,
                "realtime_factor": audio_duration / generation_time if generation_time > 0 else 0
            })
            
            # Print and save timing metrics
            print(f"  Generated in {generation_time:.2f} seconds")
            print(f"  Audio duration: {audio_duration:.2f} seconds")
            print(f"  Realtime factor: {(audio_duration / generation_time) if generation_time > 0 else 0:.2f}x")
            
            summary_file.write(f"  Generation time: {generation_time:.2f} seconds\n")
            summary_file.write(f"  Audio duration: {audio_duration:.2f} seconds\n")
            summary_file.write(f"  Realtime factor: {(audio_duration / generation_time) if generation_time > 0 else 0:.2f}x\n\n")
            
            # Save detailed profile for this generation
            save_profile_stats(
                profile, 
                f"{timestamp}_utterance_{i+1}", 
                f"Generation Profile - Utterance {i+1}",
                summary_file=summary_file
            )
            
            generated_segments.append(
                Segment(text=utterance_text, speaker=utterance_speaker, audio=audio_tensor)
            )

        # Concatenate all generations
        all_audio = torch.cat([seg.audio for seg in generated_segments], dim=0)
        torchaudio.save(
            "full_conversation.wav",
            all_audio.unsqueeze(0).cpu(),
            generator.sample_rate
        )
        
        # Print and save summary statistics
        summary_file.write("\n--- Generation Summary ---\n")
        summary_file.write(f"Total utterances: {len(conversation)}\n")
        summary_file.write(f"Total audio duration: {total_audio_duration:.2f} seconds\n")
        
        total_generation_time = sum(item["generation_time_seconds"] for item in generation_times)
        summary_file.write(f"Total generation time: {total_generation_time:.2f} seconds\n")
        summary_file.write(f"Overall realtime factor: {total_audio_duration / total_generation_time if total_generation_time > 0 else 0:.2f}x\n\n")
        
        # Print summary to console
        print("\n--- Generation Summary ---")
        print(f"Total utterances: {len(conversation)}")
        print(f"Total audio duration: {total_audio_duration:.2f} seconds")
        print(f"Total generation time: {total_generation_time:.2f} seconds")
        print(f"Overall realtime factor: {total_audio_duration / total_generation_time if total_generation_time > 0 else 0:.2f}x")
        print(f"\nAll results saved to directory: profile_results")
        print(f"Summary file: {summary_path}")


if __name__ == "__main__":
    main()
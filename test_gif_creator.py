# test_gif_creator.py
import os
import argparse
import time
from moviepy import VideoFileClip  # Standard import for VideoFileClip
from moviepy.video.fx import Resize as vfx_resize  # Standard import for resize effect


def create_optimized_gif_from_mp4(
    mp4_path,
    gif_path,
    target_mb,
    fps,
    max_width,
    content_duration_secs,
    max_loops=5,
    quality_reduction_step=0.1,  # This was okay
):  # REMOVED speed_factor from here
    if not os.path.exists(mp4_path):
        print(f"Error: MP4 file for GIF conversion not found: {mp4_path}")
        return False

    print(f"\n--- Processing MP4: {os.path.basename(mp4_path)} ---")
    print(f"Output GIF: {gif_path}")
    print(f"Target size: < {target_mb:.2f} MB")
    print(f"Initial GIF FPS: {fps}, Max Width: {max_width}px")
    print(f"Content Duration from end: {content_duration_secs}s")

    processed_clip = None

    try:
        with VideoFileClip(mp4_path) as original_clip:
            print(
                f"Original MP4 duration: {original_clip.duration:.2f}s, Original FPS: {original_clip.fps:.2f}, Original Size: {original_clip.size}"
            )

            current_clip_state = original_clip

            if original_clip.duration > content_duration_secs:
                start_time = max(0, original_clip.duration - content_duration_secs)
                print(
                    f"  Trimming original video to last {content_duration_secs}s (from {start_time:.2f}s to {original_clip.duration:.2f}s)."
                )
                current_clip_state = current_clip_state.subclip(start_time)
            else:
                print(
                    f"  Using full MP4 duration ({current_clip_state.duration:.2f}s) for GIF content."
                )

            # Speed adjustment logic has been removed.

            if current_clip_state.w > max_width:
                print(
                    f"  Resizing GIF width from {current_clip_state.w} to {max_width}."
                )
                current_clip_state = vfx_resize(  # Ensure this is the correct way to call resize from your moviepy setup
                    current_clip_state, width=max_width
                )
                print(f"  New dimensions after resize: {current_clip_state.size}")

            processed_clip = current_clip_state

            current_fps_for_gif = fps
            current_fuzz = 3
            gif_program = "ffmpeg"
            gif_generated_successfully_within_size = False

            for attempt in range(max_loops):
                print(f"\n  GIF Generation Attempt {attempt + 1}/{max_loops}:")
                print(
                    f"    Parameters: FPS={current_fps_for_gif}, Fuzz={current_fuzz}, Program={gif_program}"
                )

                # Ensure output directory for GIF exists
                output_dir_for_gif = os.path.dirname(gif_path)
                if output_dir_for_gif and not os.path.exists(
                    output_dir_for_gif
                ):  # Check if dirname is not empty
                    os.makedirs(output_dir_for_gif, exist_ok=True)

                try:
                    start_time_gif_write = time.time()
                    processed_clip.write_gif(
                        gif_path,
                        fps=current_fps_for_gif,
                        program=gif_program,
                        fuzz=current_fuzz,
                        loop=0,
                        opt="OptimizePlus" if gif_program == "gifsicle" else None,
                        logger="bar",
                    )
                    end_time_gif_write = time.time()
                    print(
                        f"    GIF write duration: {end_time_gif_write - start_time_gif_write:.2f}s"
                    )

                    if not os.path.exists(gif_path) or os.path.getsize(gif_path) == 0:
                        print("    ERROR: GIF file was not created or is empty.")
                        if gif_program == "ffmpeg":
                            print("    FFMPEG might have failed. Trying ImageMagick...")
                            gif_program = "ImageMagick"
                            current_fuzz = 5
                            current_fps_for_gif = fps  # Reset to initial
                            continue
                        else:
                            print("    ImageMagick also failed.")
                            break

                    gif_size_mb = os.path.getsize(gif_path) / (1024 * 1024)
                    print(f"    Generated GIF size: {gif_size_mb:.2f} MB")

                    if gif_size_mb <= target_mb:
                        print(f"  SUCCESS: GIF created within target size: {gif_path}")
                        gif_generated_successfully_within_size = True
                        break
                    else:
                        print("    GIF size too large. Optimizing...")
                        if current_fps_for_gif > 7 and (
                            attempt < 2 or gif_size_mb > target_mb * 1.5
                        ):
                            new_fps = max(7, int(current_fps_for_gif * 0.8))
                            print(
                                f"      Reducing FPS: {current_fps_for_gif} -> {new_fps}"
                            )
                            current_fps_for_gif = new_fps
                        elif current_fuzz < 30:
                            new_fuzz = min(30, int(current_fuzz * 1.5))
                            print(
                                f"      Increasing Fuzz: {current_fuzz} -> {new_fuzz}"
                            )
                            current_fuzz = new_fuzz
                        elif current_fps_for_gif > 5:
                            new_fps = max(5, int(current_fps_for_gif * 0.7))
                            print(
                                f"      Aggressively reducing FPS: {current_fps_for_gif} -> {new_fps}"
                            )
                            current_fps_for_gif = new_fps
                        else:
                            if gif_program == "ffmpeg":
                                print(
                                    "    FFMPEG max opt failed. Trying ImageMagick..."
                                )
                                gif_program = "ImageMagick"
                                current_fuzz = 5  # Reset for ImageMagick
                                current_fps_for_gif = fps // 2  # Reset for ImageMagick
                            else:
                                print(f"    Could not reduce GIF below {target_mb}MB.")
                                break
                except Exception as e_gif:
                    print(
                        f"    ERROR during GIF generation with {gif_program}: {e_gif}"
                    )
                    if gif_program == "ffmpeg":
                        print("    FFMPEG error. Trying ImageMagick...")
                        gif_program = "ImageMagick"
                        current_fuzz = 5
                        current_fps_for_gif = fps // 2
                    else:
                        print(
                            "    GIF generation failed with errors using all programs."
                        )
                        break

            if not gif_generated_successfully_within_size:
                if (
                    os.path.exists(gif_path)
                    and (os.path.getsize(gif_path) / (1024 * 1024)) > target_mb
                ):
                    print(
                        f"  WARNING: Final GIF size still exceeds target. Kept at: {gif_path}"
                    )
                elif not os.path.exists(gif_path):
                    print("  FAILURE: GIF not created.")
            return os.path.exists(gif_path) and os.path.getsize(gif_path) > 0

    except Exception as e:
        print(f"An critical error occurred processing {mp4_path}: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        # original_clip closed by 'with'.
        # processed_clip is the one that might need closing if it's a VideoFileClip instance
        # and distinct from original_clip (which it will be if any transformation is applied).
        if (
            processed_clip is not None
            and hasattr(processed_clip, "close")
            and callable(processed_clip.close)
        ):
            # Check if it's actually a VideoFileClip and has an open reader
            if (
                isinstance(processed_clip, VideoFileClip)
                and hasattr(processed_clip, "reader")
                and processed_clip.reader
            ):
                try:
                    processed_clip.close()
                except Exception:
                    pass  # Ignore errors on this best-effort close


def main():
    parser = argparse.ArgumentParser(
        description="Convert a specific MP4 video to an optimized GIF."
    )
    parser.add_argument("input_mp4", type=str, help="Path to the input MP4 file.")
    parser.add_argument(
        "output_gif", type=str, help="Path to save the output GIF file."
    )
    parser.add_argument(
        "--target_mb", type=float, default=48.0, help="Target maximum GIF size in MB."
    )
    parser.add_argument(
        "--max_width", type=int, default=320, help="Maximum width of the output GIF."
    )
    parser.add_argument("--fps", type=int, default=10, help="Target FPS for the GIF.")
    # No --speed argument here
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Max duration (original video seconds, from end) for GIF.",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.input_mp4) or not args.input_mp4.lower().endswith(
        ".mp4"
    ):
        print(f"Error: Input path '{args.input_mp4}' is not a valid MP4 file.")
        return

    output_dir = os.path.dirname(args.output_gif)
    if output_dir and not os.path.exists(output_dir):  # Check if dirname is not empty
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    success = create_optimized_gif_from_mp4(
        args.input_mp4,
        args.output_gif,
        target_mb=args.target_mb,
        fps=args.fps,
        max_width=args.max_width,
        # speed_factor=1.0, # REMOVED from call site
        content_duration_secs=args.duration,
    )

    if success and os.path.exists(args.output_gif):
        print(f"\nGIF processing complete. Output at: {args.output_gif}")
    else:
        print(
            f"\nGIF processing failed or did not produce a valid output file at: {args.output_gif}"
        )


if __name__ == "__main__":
    main()

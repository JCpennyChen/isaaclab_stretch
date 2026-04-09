import os
import numpy as np
import imageio


class VideoRecorder:
    """Video recorder using omni.replicator annotator for Isaac Sim."""

    def __init__(self, video_dir, fps=30):
        self.video_dir = video_dir
        self.fps = fps
        self.frames = []
        os.makedirs(video_dir, exist_ok=True)

        import omni.replicator.core as rep
        from omni.kit.viewport.utility import get_active_viewport

        self.viewport = get_active_viewport()
        self.render_product = rep.create.render_product(
            self.viewport.get_active_camera(),
            resolution=(1280, 720),
        )
        self.rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self.rgb_annotator.attach([self.render_product])

    def capture_frame(self):
        """Capture the current frame via replicator annotator."""
        try:
            # Flush the render pipeline so the annotator has fresh data.
            # Do NOT use rep.orchestrator.step() — it fights IsaacLab's
            # own timeline and deadlocks in headless mode.
            import omni.kit.app

            omni.kit.app.get_app().update()

            data = self.rgb_annotator.get_data()
            if data is not None and data.size > 0:
                frame = np.array(data)
                if frame.ndim == 3:
                    # Drop alpha channel if present (RGBA -> RGB)
                    if frame.shape[2] == 4:
                        frame = frame[:, :, :3]
                    self.frames.append(frame)
        except Exception as e:
            if len(self.frames) == 0:
                print(f"[Video] Warning: frame capture failed: {e}")

    def save(self, filename):
        """Save captured frames to an mp4 file."""
        if len(self.frames) == 0:
            print("[Video] No frames captured, skipping save.")
            return None

        filepath = os.path.join(self.video_dir, filename)
        print(f"[Video] Saving {len(self.frames)} frames to {filepath}")
        imageio.mimwrite(filepath, self.frames, fps=self.fps)
        self.frames = []
        return filepath

    def reset(self):
        """Clear captured frames."""
        self.frames = []

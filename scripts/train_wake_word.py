"""
Train a custom "EV" wake word model for openwakeword.

Usage:
    python -m scripts.train_wake_word

Generates synthetic audio using espeak-ng, computes openwakeword features,
trains a small DNN, and exports an ONNX model.
"""
import os
import subprocess
import tempfile
import uuid
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import torch

SAMPLE_RATE = 16000
OUTPUT_DIR = Path("models/wake_word")
CLIP_DURATION_SAMPLES = 32000  # 2 seconds
N_INPUT_FRAMES = 16


def generate_espeak_clip(text, voice="en-us", speed=160, pitch=50, amplitude=100):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    subprocess.run(
        ["espeak-ng", "-v", voice, "-s", str(speed), "-p", str(pitch),
         "-a", str(amplitude), "-w", tmp_path, text],
        check=True, capture_output=True,
    )
    sr, data = scipy.io.wavfile.read(tmp_path)
    os.unlink(tmp_path)
    if data.dtype != np.int16:
        data = (data * 32767).astype(np.int16)
    if sr != SAMPLE_RATE:
        n_samples = int(len(data) * SAMPLE_RATE / sr)
        data = np.interp(
            np.linspace(0, len(data), n_samples),
            np.arange(len(data)),
            data.astype(np.float32),
        ).astype(np.int16)
    return data


def pad_or_trim(audio, length=CLIP_DURATION_SAMPLES):
    if len(audio) >= length:
        return audio[:length]
    padded = np.zeros(length, dtype=audio.dtype)
    start = np.random.randint(0, length - len(audio))
    padded[start:start + len(audio)] = audio
    return padded


def add_noise(audio, noise_level=0.005):
    noise = np.random.randn(len(audio)) * noise_level * 32767
    return np.clip(audio.astype(np.float32) + noise, -32768, 32767).astype(np.int16)


def generate_clips(phrases, output_dir, n_target=500, label=""):
    output_dir.mkdir(parents=True, exist_ok=True)
    voices = ["en-us", "en-gb", "en-gb-scotland", "en-gb-x-rp", "en-029"]
    speeds = range(120, 220, 15)
    pitches = range(20, 80, 10)

    clips = []
    for phrase in phrases:
        for voice in voices:
            for speed in speeds:
                for pitch in pitches:
                    if len(clips) >= n_target:
                        break
                    try:
                        audio = generate_espeak_clip(phrase, voice=voice, speed=speed, pitch=pitch)
                        audio = pad_or_trim(audio)
                        audio = add_noise(audio, noise_level=np.random.uniform(0.001, 0.02))
                        clips.append(audio)
                        path = output_dir / f"{uuid.uuid4().hex}.wav"
                        scipy.io.wavfile.write(str(path), SAMPLE_RATE, audio)
                    except Exception:
                        continue
                if len(clips) >= n_target:
                    break
            if len(clips) >= n_target:
                break
        if len(clips) >= n_target:
            break

    print(f"  Generated {len(clips)} {label} clips")
    return clips


def clips_to_features(clips):
    import openwakeword
    oww = openwakeword.Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
    model_name = "hey_jarvis"

    features = []
    for clip in clips:
        clip_int16 = clip.astype(np.int16) if clip.dtype != np.int16 else clip
        oww.reset()

        step = 1280
        for i in range(0, len(clip_int16) - step + 1, step):
            chunk = clip_int16[i:i + step]
            oww.predict(chunk)

        feat = oww.preprocessor.get_features(N_INPUT_FRAMES)
        features.append(feat.squeeze(0))

    return np.array(features)


class WakeWordDNN(torch.nn.Module):
    def __init__(self, input_shape, layer_dim=128):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(input_shape[0] * input_shape[1], layer_dim),
            torch.nn.LayerNorm(layer_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(layer_dim, layer_dim),
            torch.nn.LayerNorm(layer_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(layer_dim, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


def train_model(positive_features, negative_features, output_dir):
    n_pos = len(positive_features)
    n_neg = len(negative_features)

    X = np.concatenate([positive_features, negative_features])
    y = np.array([1] * n_pos + [0] * n_neg, dtype=np.float32)

    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    split = int(0.85 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    input_shape = (X.shape[1], X.shape[2])
    print(f"  Feature shape: {input_shape}")
    print(f"  Training: {split}, Validation: {len(X) - split}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WakeWordDNN(input_shape).to(device)

    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()

    best_val_loss = float("inf")
    best_state = None
    patience = 20
    no_improve = 0

    for epoch in range(200):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).squeeze()
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).squeeze()
                val_loss += criterion(pred, yb).item()
                correct += ((pred > 0.5).float() == yb).sum().item()
                total += len(yb)

        val_loss /= len(val_loader)
        acc = correct / total

        if epoch % 10 == 0:
            print(f"    Epoch {epoch:3d}: train_loss={train_loss / len(train_loader):.4f}  val_loss={val_loss:.4f}  val_acc={acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model.eval()

    onnx_path = output_dir / "ev_wakeword.onnx"
    dummy = torch.randn(1, input_shape[0], input_shape[1]).to(device)
    torch.onnx.export(
        model.cpu(), dummy.cpu(), str(onnx_path),
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    print(f"  Model exported to {onnx_path}")
    return str(onnx_path)


REAL_POSITIVE_DIR = OUTPUT_DIR / "clips" / "real_positive"


def load_real_clips(directory: Path) -> list:
    clips = []
    wav_files = sorted(directory.glob("*.wav")) if directory.exists() else []
    for wav_path in wav_files:
        try:
            sr, data = scipy.io.wavfile.read(str(wav_path))
            if data.dtype != np.int16:
                data = (data * 32767).astype(np.int16)
            if sr != SAMPLE_RATE:
                n_samples = int(len(data) * SAMPLE_RATE / sr)
                data = np.interp(
                    np.linspace(0, len(data), n_samples),
                    np.arange(len(data)),
                    data.astype(np.float32),
                ).astype(np.int16)
            if data.ndim > 1:
                data = data[:, 0]
            clips.append(pad_or_trim(data))
        except Exception as e:
            print(f"    Skipping {wav_path.name}: {e}")
    return clips


def augment_clip(audio: np.ndarray, n_variants: int = 3) -> list:
    """Return n_variants augmented copies of a real recording."""
    variants = [audio]
    for _ in range(n_variants - 1):
        noise_level = np.random.uniform(0.001, 0.015)
        aug = add_noise(audio, noise_level)
        # slight volume jitter
        scale = np.random.uniform(0.8, 1.2)
        aug = np.clip(aug.astype(np.float32) * scale, -32768, 32767).astype(np.int16)
        variants.append(aug)
    return variants


def main():
    print("=" * 50)
    print("  EV — Wake Word Training")
    print("=" * 50)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pos_dir = OUTPUT_DIR / "clips" / "positive"
    neg_dir = OUTPUT_DIR / "clips" / "negative"

    positive_phrases = ["E V", "E. V.", "eevee", "E.V."]
    negative_phrases = [
        "easy", "every", "even", "heavy", "levy", "TV", "evening",
        "evil", "Eva", "event", "ever", "Elvis", "envy", "evade",
        "hello", "hey", "okay", "sorry", "what", "please", "thanks",
        "listen", "open", "close", "start", "stop", "play", "pause",
        "the weather is nice", "turn on the lights",
        "what time is it", "how are you",
    ]

    print("\n[1/5] Loading real voice recordings...")
    real_clips = load_real_clips(REAL_POSITIVE_DIR)
    if real_clips:
        augmented = []
        for clip in real_clips:
            augmented.extend(augment_clip(clip, n_variants=5))
        print(f"  Loaded {len(real_clips)} real clips → {len(augmented)} with augmentation")
    else:
        augmented = []
        print("  No real recordings found. Run `python -m scripts.record_wake_word` first for best results.")

    print("\n[2/5] Generating synthetic positive clips ('EV')...")
    n_synthetic = max(100, 500 - len(augmented))
    pos_clips = generate_clips(positive_phrases, pos_dir, n_target=n_synthetic, label="positive")
    # real clips go first so they're weighted by repetition
    pos_clips = augmented + pos_clips

    print("[3/5] Generating negative clips...")
    neg_clips = generate_clips(negative_phrases, neg_dir, n_target=500, label="negative")

    print("[4/5] Computing openwakeword features...")
    pos_features = clips_to_features(pos_clips)
    neg_features = clips_to_features(neg_clips)
    print(f"  Positive features: {pos_features.shape}")
    print(f"  Negative features: {neg_features.shape}")

    print("[5/5] Training wake word model...")
    onnx_path = train_model(pos_features, neg_features, OUTPUT_DIR)

    print(f"\nDone! Wake word model saved to: {onnx_path}")
    print("Updating detector to use the new model...")


if __name__ == "__main__":
    main()

from pathlib import Path
import math
import torch
from model import OccupancyCalc


checkpoint_path = Path(__file__).resolve().parent / "occupancy_model.pth"

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True
)

model = OccupancyCalc()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

mean = checkpoint["mean"]
std = checkpoint["std"]


def read_sensor(prompt, minimum, maximum=None):
    while True:
        try:
            value = float(input(prompt))
        except ValueError:
            print("Geçerli bir sayı girin (ondalık ayırıcı: nokta).")
            continue
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("\nGiriş iptal edildi.")

        if not math.isfinite(value):
            print("NaN veya sonsuzluk yerine sonlu bir sayı girin.")
        elif value < minimum or (maximum is not None and value > maximum):
            upper = f" ve en fazla {maximum}" if maximum is not None else ""
            print(f"Değer en az {minimum}{upper} olmalıdır.")
        elif value > torch.finfo(torch.float32).max:
            print("Değer modelin sayısal aralığı için çok büyük.")
        else:
            return value


temperature = read_sensor("Temperature (°C): ", -273.15)
humidity = read_sensor("Humidity (%): ", 0, 100)
light = read_sensor("Light (lux): ", 0)
co2 = read_sensor("CO2 (ppm): ", 0, 1_000_000)
humidity_ratio = read_sensor("Humidity Ratio: ", 0)


sensor = torch.tensor([
    [temperature, humidity, light, co2, humidity_ratio]
], dtype=torch.float32)


sensor_scaled = (sensor - mean) / std


with torch.inference_mode():
    logit = model(sensor_scaled)
    if not torch.isfinite(logit).all():
        raise SystemExit("Bu değerlerle sonlu bir tahmin hesaplanamadı; sensör girdilerini kontrol edin.")
    probability = torch.sigmoid(logit).item()


prediction = 1 if probability >= 0.5 else 0

print(f"\nDoluluk Olasılığı: %{probability * 100:.2f}")

if prediction == 1:
    print("Tahmin: ODA DOLU")
else:
    print("Tahmin: ODA BOŞ")
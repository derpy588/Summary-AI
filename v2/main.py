from models import T100

model = T100()

out = model.evaluate("This is a test.")

model.train_file("training_data/cnn-daily-03.parquet", "article", "highlights", 5, 32, True, 0.0001)

model.save("T100-pickle.pt")
model.save("T100-safetensors.safetensors")


from src.preprocess import main as preprocess_main
from src.train import train as train_main
from src.melodygenerator import MelodyGenerator, SEQUENCE_LENGTH
import tensorflow as tf


physical_devices = tf.config.list_physical_devices("GPU")
print("TensorFlow number of gpus:", physical_devices)
if __name__ == "__main__":
    preprocess_main()
    train_main()
    mg = MelodyGenerator()
    seed = "67 _ 67 _ 67 _ _ 65 64 _ 64 _ 64 _ _"
    seed2 = "67 _ _ _ _ _ 65 _ 64 _ 62 _ 60 _ _ _"
    melody = mg.generate_melody(seed, 500, SEQUENCE_LENGTH, 0.3)
    print(melody)
    mg.save_melody(melody)
    # Visit https://midiplayer.ehubsoft.net/ to play the generated melody.

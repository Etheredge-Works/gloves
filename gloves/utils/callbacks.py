from tensorflow.keras.callbacks import Callback
import dvclive
import mlflow
import psutil


class MetricsCallback(Callback):
    def on_epoch_end(self, epoch: int, logs: dict = None):
        logs = logs or {}
        for metric, value in logs.items():
            float_value = value
            if type(value) == np.float32:
                float_value = float(value)
            dvclive.log(metric, float_value)

        mem = psutil.virtual_memory().used/8/1024/1024/1024
        dvclive.log('memory_use_GB', mem)
        mlflow.log_metric('memory_use_GB', mem)

        dvclive.next_step()

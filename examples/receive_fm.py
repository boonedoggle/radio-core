#!/usr/bin/env python3
import sys
import time
import queue
import sounddevice as sd
from threading import Thread
from dataclasses import dataclass
from argparse import ArgumentParser

from SoapySDR import Device, SOAPY_SDR_CF32, SOAPY_SDR_RX
from radiocore import Buffer, RingBuffer, FM, MFM, WBFM, Decimate

# =============================================================================
# Global Defaults
# =============================================================================
ENABLE_CUDA: bool = False
FREQUENCY: float = 94.1e6
OFFSET_FREQUENCY: float = 312e6
DEEMPHASIS: float = 75e-6
CLOCK_RATE: float = 61.44e6
INPUT_RATE: float = CLOCK_RATE / 32
DEMOD_RATE: float = 250e3
AUDIO_RATE: float = 48e3
DEVICE_NAME: str = "SoapyAIRT"
DEMODULATOR: str = FM


def parse_args():
    parser = ArgumentParser(description="FM Receiver and Demodulator")
    parser.add_argument("--enable-cuda", action="store_true",
                        help="Enable CUDA demodulation")
    parser.add_argument("--offset-frequency", type=float, default=OFFSET_FREQUENCY,
                        help=f"Frequency offset if using an upconverter in Hz (default: {OFFSET_FREQUENCY})")
    parser.add_argument("--frequency", type=float, default=FREQUENCY,
                        help=f"Set the FM station frequency in Hz (default: {FREQUENCY})")
    parser.add_argument("--deemphasis", type=float, default=DEEMPHASIS,
                        help=f"75e-6 for Americas and Korea, otherwise 50e-6 (default: {DEEMPHASIS})")
    parser.add_argument("--clock-rate", type=float, default=CLOCK_RATE,
                        help=f"AIR-T master clock rate in Hz (default: {CLOCK_RATE})")
    parser.add_argument("--input-rate", type=float, default=INPUT_RATE,
                        help=f"AIR-T sample rate in Hz (default: {INPUT_RATE})")
    parser.add_argument("--demod-rate", type=float, default=DEMOD_RATE,
                        help=f"FM station bandwidth. (240-256 kHz) in Hz (default: {DEMOD_RATE})")
    parser.add_argument("--audio-rate", type=float, default=AUDIO_RATE,
                        help=f"Audio output sample rate in Hz (default: {AUDIO_RATE})")
    parser.add_argument("--device-name", type=str, default=DEVICE_NAME,
                        help=f"Directory to save output files (default: '{DEVICE_NAME}')")
    parser.add_argument("--demodulator", default=DEMODULATOR,
                        help=f"Demodulator (WBFM, MFM, or FM) (default: '{DEMODULATOR}')")
    return parser.parse_args()


class SdrDevice(Thread):
    def __init__(self, _config):
        super().__init__()
        self._config = _config
        self.running = False

        print("Configuring SDR device...")
        self.sdr = Device({"driver": self._config.device_name,
                           "master_clock_rate": str(self._config.clock_rate)})
        self.sdr.setSampleRate(SOAPY_SDR_RX, 0, self._config.input_rate)
        self.sdr.setFrequency(SOAPY_SDR_RX, 0, self._config.frequency + self._config.offset_frequency)
        self.sdr.setGainMode(SOAPY_SDR_RX, 0, True)
        self.rx = self.sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)

        print("Allocating SDR device buffers...")
        self.buffer = RingBuffer(self._config.input_rate * 3,
                                 cuda=self._config.enable_cuda)

    @property
    def output(self) -> RingBuffer:
        return self.buffer

    def run(self):
        tmp_buffer = Buffer(2**16, cuda=self._config.enable_cuda)

        self.sdr.activateStream(self.rx)
        self.running = True

        while self.running:
            c = self.sdr.readStream(self.rx,
                                    [tmp_buffer.data],
                                    tmp_buffer.size,
                                    timeoutUs=500000)
            if c.ret > 0:
                self.buffer.put(tmp_buffer.data[:c.ret])

    def stop(self):
        self.sdr.deactivateStream(self.rx)
        self.sdr.closeStream(self.rx)
        self.running = False
        self.join()


class Dsp(Thread):

    def __init__(self, _config, data_in: RingBuffer):
        super().__init__()
        self._config = _config
        self.data_in = data_in
        self.running = False

        print("Configuring DSP...")
        demod = eval(self._config.demodulator)
        self.demod = demod(self._config.demod_rate,
                           self._config.audio_rate,
                           deemphasis=self._config.deemphasis,
                           cuda=self._config.enable_cuda)
        self.decim = Decimate(self._config.input_rate,
                              self._config.demod_rate,
                              cuda=self._config.enable_cuda)

        print("Allocating DSP buffers...")
        self.que = queue.Queue()

    @property
    def output(self) -> queue.Queue:
        return self.que

    def run(self):
        tmp_buffer = Buffer(self._config.input_rate, cuda=self._config.enable_cuda)

        self.running = True

        while self.running:
            if not self.data_in.get(tmp_buffer.data):
                continue

            tmp = self.decim.run(tmp_buffer.data)
            tmp = self.demod.run(tmp)

            self.que.put_nowait(tmp)

    def stop(self):
        self.running = False
        self.join()


if __name__ == "__main__":
    config = parse_args()

    # Configure SDR device thread.
    rx = SdrDevice(config)
    dsp = Dsp(config, rx.output)

    # Define demodulation callback. This should not block.
    def process(outdata, *_):
        if not dsp.output.empty():
            outdata[:] = dsp.output.get_nowait()
        else:
            outdata[:] = 0.0

    # Configure sound device stream.
    stream = sd.OutputStream(blocksize=int(config.audio_rate),
                             callback=process,
                             samplerate=int(config.audio_rate),
                             channels=dsp.demod.channels)

    try:
        print("Starting playback...")
        rx.start()
        dsp.start()
        stream.start()

        # Busy loop until interrupted by KeyboardInterrupt.
        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:
        dsp.stop()
        rx.stop()
        sys.exit('\nInterrupted by user. Closing...')

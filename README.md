# 📡 5G-NR Wireless Communication Simulator

**A Modular PHY Layer Simulation Framework for 5G New Radio**

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-orange)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-green)
![Flask](https://img.shields.io/badge/Flask-WebApp-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

Models the end-to-end 5G physical layer transmission chain — from random bit generation through modulation, OFDM, MIMO, channel fading, equalization, and performance analysis. Produces publication-quality visualizations and includes an interactive web dashboard.

---

## 🚀 Quick Start

```bash
# Install
pip install -r requirements.txt

# Run experiments
python main.py --experiment 1    # Modulation comparison
python main.py --experiment 2    # Channel comparison
python main.py --experiment 3    # OFDM with equalization
python main.py --experiment 4    # Advanced features demo

# Launch web dashboard
cd webapp && python app.py       # Opens at http://localhost:5000

# Or Streamlit dashboard
streamlit run dashboard.py
```

---

## 🎯 Features

### Core PHY Layer
| Feature | Details |
|---------|---------|
| **Modulation** | BPSK, QPSK, 16-QAM, 64-QAM, 256-QAM with Gray coding |
| **Channel Models** | AWGN, Rayleigh (flat & freq-selective), Rician (K-factor), Doppler (Jakes spectrum) |
| **OFDM** | FFT/IFFT, cyclic prefix, subcarrier mapping, resource grid, pilot insertion |
| **MIMO** | 2×2 spatial multiplexing, Alamouti STBC, receive diversity (MRC/EGC/SC) |
| **Equalization** | Zero Forcing, MMSE, ML detection |
| **Channel Estimation** | LS, MMSE, Linear interpolation from pilots |

### Advanced 5G NR Features
| Feature | Details |
|---------|---------|
| **LDPC Coding** | Repeat-Accumulate code + Min-sum Belief Propagation decoder |
| **AMC** | Adaptive Modulation & Coding — auto-selects scheme per SNR |
| **Beamforming** | ULA steering vectors, beamscan DOA, LCMV beamformer |
| **MU-MIMO** | ZF, MMSE, Block Diagonalization precoding with sum rate analysis |
| **HARQ** | Chase combining, Incremental Redundancy, multi-process controller |
| **Numerology** | μ=0..4 (15/30/60/120/240 kHz SCS), slot/frame structure, resource grid |

### Performance Analysis
- **BER / SER** vs SNR for any modulation × channel combination
- **Throughput** — effective data rate accounting for errors & overhead
- **Spectral Efficiency** — bps/Hz with Shannon capacity comparison
- **Latency** — end-to-end delay including retransmission modeling

---

## 📦 Installation

### Prerequisites
- Python 3.12 or higher
- pip package manager

### Windows
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### Linux / macOS
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Verify Installation
```bash
python -c "import numpy, scipy, matplotlib; print('All dependencies OK')"
```

---

## 🏗️ Project Structure

```
5G-NR-Simulator/
│
├── main.py                     # CLI entry point
├── config.py                   # Centralized simulation parameters
├── simulation_controller.py    # Orchestrates TX → Channel → RX pipeline
├── dashboard.py                # Streamlit interactive dashboard
├── requirements.txt
│
├── transmitter/                # TX chain
│   ├── bit_generator.py        # Random/packet bit generation
│   ├── modulation.py           # BPSK, QPSK, 16/64/256-QAM with Gray coding
│   ├── coding.py               # Convolutional encoder with puncturing
│   ├── mapper.py               # OFDM resource grid mapping with pilots
│   └── adaptive.py             # SNR-based Adaptive Modulation & Coding
│
├── receiver/                   # RX chain
│   ├── demodulation.py         # Soft/hard demapper with LLR computation
│   ├── decoder.py              # Viterbi decoder
│   ├── equalizer.py            # Zero Forcing & MMSE equalization
│   ├── detector.py             # ML, ZF, MMSE symbol detection
│   ├── channel_estimation.py   # Pilot-based LS/MMSE/Linear estimation
│   └── harq.py                 # Chase combining & Incremental Redundancy
│
├── channel/                    # Wireless channel models
│   ├── awgn.py                 # Additive White Gaussian Noise
│   ├── rayleigh.py             # Flat & frequency-selective Rayleigh fading
│   ├── rician.py               # Rician fading with LOS component
│   ├── doppler.py              # Jakes Doppler spectrum simulator
│   └── channel_models.py       # Factory pattern for channel creation
│
├── ofdm/                       # OFDM processing
│   ├── transmitter.py          # OFDM modulator (IFFT + subcarrier mapping)
│   ├── receiver.py             # OFDM demodulator (FFT + subcarrier demapping)
│   ├── cyclic_prefix.py        # CP insertion & removal
│   └── fft_utils.py            # FFT/IFFT wrapper with frequency bins
│
├── mimo/                       # MIMO processing
│   ├── mimo2x2.py              # 2×2 MIMO with ZF/MMSE detectors
│   ├── alamouti.py             # Alamouti STBC encoder & decoder
│   ├── diversity.py            # Selection, Equal Gain, Maximal Ratio combining
│   ├── beamforming.py          # ULA steering, beamscan, MMSE/LCMV beamformers
│   └── mumimo.py               # MU-MIMO with ZF/MMSE/BD precoding & sum rate
│
├── numerology/                 # 5G NR numerology
│   ├── spacing.py              # Subcarrier spacing (15-240 kHz), slot timing
│   └── frame_structure.py      # Frame/subframe/slot/symbol structure
│
├── coding/                     # Channel coding
│   └── ldpc.py                 # RA-LDPC encoder + Min-sum BP decoder
│
├── analysis/                   # Performance analysis
│   ├── ber.py                  # BER/SER calculator with Monte Carlo & theoretical
│   ├── throughput.py           # Data rate & throughput calculation
│   ├── latency.py              # End-to-end latency with retransmissions
│   └── spectral_efficiency.py  # bps/Hz with Shannon limit comparison
│
├── visualization/              # Plotting
│   ├── constellation.py        # Constellation diagrams (ideal & noisy)
│   ├── ber_plot.py             # BER vs SNR curves (semilogy)
│   ├── channel_plot.py         # Frequency/impulse/Doppler response
│   └── spectrum.py             # Power spectrum, OFDM spectrum, waterfall
│
├── utils/                      # Utilities
│   ├── helpers.py              # Gray coding, energy computation
│   └── logger.py               # Logging configuration
│
├── experiments/                # Pre-built experiment scripts
│   ├── experiment1.py          # Modulation BER comparison
│   ├── experiment2.py          # Channel model comparison
│   ├── experiment3.py          # OFDM equalizer comparison
│   └── experiment4.py          # Advanced features demo
│
├── webapp/                     # Flask web application
│   ├── app.py                  # Flask server with 9 API endpoints
│   └── templates/
│       └── index.html          # Dark-theme Bootstrap 5 dashboard
│
├── reports/                    # Generated output graphs
└── docs/                       # Documentation
```

---

## 📚 Module Deep Dive

### Transmitter Chain
```
Random Bits → [Bit Generator] → [Encoder] → [Modulator] → [Resource Mapper] → OFDM
```

### Channel Models
```
      ┌─────────┐
      │  AWGN   │ ← Thermal noise, simplest model
      ├─────────┤
      │Rayleigh │ ← Multipath fading, no LOS (flat or freq-selective)
      ├─────────┤
      │ Rician  │ ← Fading with dominant LOS path (K-factor controls LOS strength)
      ├─────────┤
      │ Doppler │ ← Time-varying fading (Jakes spectrum, controlled by Doppler freq)
      └─────────┘
```

### Receiver Chain
```
OFDM → [Channel Estimator] → [Equalizer] → [Detector] → [Demodulator] → [Decoder] → BER
                                              ↑
                                        HARQ Buffer (Chase Combine)
```

---

## 🧪 Advanced Features

### LDPC Coding
Repeat-Accumulate code with min-sum belief propagation decoder. Corrects bit errors using iterative message passing on a Tanner graph (coding gain ~3 dB).

### Adaptive Modulation & Coding (AMC)
Auto-selects modulation scheme based on estimated SNR:
| SNR Range | Scheme | Throughput |
|-----------|--------|------------|
| < 3 dB | BPSK | 1 bps/Hz |
| 3-9 dB | QPSK | 2 bps/Hz |
| 9-17 dB | 16QAM | 4 bps/Hz |
| 17-23 dB | 64QAM | 6 bps/Hz |
| > 23 dB | 256QAM | 8 bps/Hz |

### Beamforming
Uniform Linear Array (ULA) with adjustable element count. Supports steering vector beamforming, beamscan direction-of-arrival estimation, and MMSE/LCMV adaptive beamformers.

### MU-MIMO Precoding
Multi-user MIMO with three precoding strategies:
- **ZF**: Zero Forcing — cancels inter-user interference
- **MMSE**: Minimum Mean Square Error — balances interference & noise
- **BD**: Block Diagonalization — nulls interference per user group

### HARQ
Hybrid Automatic Repeat Request with:
- **Chase Combining**: Retransmit same data, combine with MRC
- **Incremental Redundancy**: Different parity versions per retransmission
- Multi-process controller supporting up to 8 parallel HARQ processes

---

## 💻 Usage

### Command Line Interface
```bash
# Full argument list
python main.py --help

# Custom simulation
python main.py --modulation 64QAM --channel Rayleigh --num-bits 100000 --snr-range 5,35,3

# Full analysis with all metrics
python main.py --full-analysis --modulation QPSK --channel AWGN --snr-range 0,26,2

# Constellation + numerology
python main.py --constellation --modulation 16QAM --numerology 1
```

### Experiments
```bash
# Experiment 1: BER vs SNR for BPSK/QPSK/16QAM/64QAM (AWGN)
python experiments/experiment1.py

# Experiment 2: BER across AWGN vs Rayleigh vs Rician (QPSK)
python experiments/experiment2.py

# Experiment 3: OFDM with ZF vs MMSE vs No equalization
python experiments/experiment3.py

# Experiment 4: All advanced features (LDPC, AMC, Beamforming, MU-MIMO, HARQ)
python experiments/experiment4.py
```

### Web Dashboard
```bash
cd webapp
python app.py
# Open http://localhost:5000 in your browser
```
The web app provides 8 interactive tabs: BER Curves, Constellation, OFDM Spectrum, MU-MIMO, Beamforming, Full Analysis, LDPC Coding, and Numerology.

---

## 📊 5G NR Numerologies

| μ | SCS (kHz) | Slot Duration | Sampling Rate | Bandwidth (100 RB) | Use Case |
|---|-----------|---------------|---------------|-------------------|----------|
| 0 | 15 | 1 ms | 7.68 MHz | 18 MHz | Wide area coverage, sub-6 GHz |
| 1 | 30 | 0.5 ms | 15.36 MHz | 36 MHz | Urban macro, sub-6 GHz |
| 2 | 60 | 0.25 ms | 30.72 MHz | 72 MHz | mmWave FR2, low latency |
| 3 | 120 | 0.125 ms | 61.44 MHz | 144 MHz | High mobility, mmWave |
| 4 | 240 | 0.0625 ms | 122.88 MHz | 288 MHz | Specialized, extreme throughput |

**Trade-offs:**
- Lower μ → Better coverage, more robust to delay spread
- Higher μ → Lower latency, better Doppler tolerance, higher throughput

---

## 📈 Key Performance Metrics

| Metric | Description |
|--------|-------------|
| **Bit Error Rate (BER)** | Ratio of bit errors to total transmitted bits |
| **Symbol Error Rate (SER)** | Ratio of symbol errors to total transmitted symbols |
| **Throughput** | Effective data rate after accounting for errors |
| **Spectral Efficiency** | Throughput per unit bandwidth (bps/Hz) |
| **Latency** | End-to-end delay including retransmission delays |
| **PAPR** | Peak-to-Average Power Ratio in OFDM signals |
| **Sum Rate** | Total achievable rate in MU-MIMO systems |

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python 3.12+ | Core language |
| NumPy | Numerical computing, vectorized operations |
| SciPy | Signal processing, sparse matrices |
| Matplotlib | Publication-quality visualizations |
| Flask | Web application framework |
| Streamlit | Interactive dashboard (alternative) |
| pytest | Unit testing |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

See the `docs/` folder for architecture details and coding conventions.

---

## 📜 License

MIT © Sohan

---

## 📬 Contact

Project Link: [https://github.com/augsatr/5G-NR-Wireless-Communication-Simulator](https://github.com/augsatr/5G-NR-Wireless-Communication-Simulator)

---

## 🙏 Acknowledgments

- 3GPP TS 38.211 — NR Physical Channels and Modulation
- 3GPP TS 38.214 — NR Physical Layer Procedures for Data
- David Tse & Pramod Viswanath — Fundamentals of Wireless Communication
- Thomas M. Cover & Joy A. Thomas — Elements of Information Theory

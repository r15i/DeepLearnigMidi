import torch
import numpy as np
from scipy.stats import wasserstein_distance
from dtw import dtw  

class MusicEvaluator:
    def __init__(self, model, device, threshold=0.5):
        """
        model: il tuo VAE (ConvVAE, ResVAE, ...)
        device: torch.device
        threshold: soglia per binarizzare il piano roll
        """
        self.model = model.to(device)
        self.device = device
        self.threshold = threshold

    # ===================== UTILS ===================== #
    def _binarize(self, x_logits):
        """Applica sigmoid e threshold → binario (0/1)."""
        probs = torch.sigmoid(x_logits)
        return (probs > self.threshold).float()

    def _pitch_histogram(self, roll):
        """
        roll: tensor (B, 1, 128, 16) binario
        ritorna: distribuzione media di pitch (128,)
        """
        roll = roll.squeeze(1).cpu().numpy()  # (B, 128, 16)
        hist = np.sum(roll, axis=2)           # conta ON lungo il tempo → (B, 128)
        hist = hist / (np.sum(hist, axis=1, keepdims=True) + 1e-8)  # normalizza per sample
        return np.mean(hist, axis=0)          # media sui batch

    def _rhythm_histogram(self, roll):
        """
        Estrae distribuzione delle durate ON consecutive.
        roll: (B, 1, 128, 16)
        ritorna: istogramma normalizzato delle durate
        """
        roll = roll.squeeze(1).cpu().numpy()  # (B,128,16)
        durations = []
        for seq in roll:  # (128,16)
            for note_seq in seq:  # (16,)
                length = 0
                for val in note_seq:
                    if val == 1:
                        length += 1
                    elif length > 0:
                        durations.append(length)
                        length = 0
                if length > 0:
                    durations.append(length)
        if len(durations) == 0:
            return np.zeros(16)
        hist, _ = np.histogram(durations, bins=np.arange(1, 18))  # max durata=16
        return hist / np.sum(hist)

    # ===================== METRICS ===================== #
    def pitch_distribution_similarity(self, real, gen):
        """Wasserstein distance tra distribuzioni di pitch."""
        real_hist = self._pitch_histogram(real)
        gen_hist = self._pitch_histogram(gen)
        return wasserstein_distance(real_hist, gen_hist)

    def rhythm_similarity(self, real, gen):
        """Wasserstein distance fra istogrammi di durate."""
        real_hist = self._rhythm_histogram(real)
        gen_hist = self._rhythm_histogram(gen)
        return wasserstein_distance(real_hist, gen_hist)

    def dtw_distance(self, real, gen):
        """
        Calcola DTW medio fra sequenze flattenate (pitch-time).
        real, gen: tensor (B,1,128,16)
        """
        real = real.squeeze(1).cpu().numpy()  # (B,128,16)
        gen = gen.squeeze(1).cpu().numpy()
        dists = []
        for r, g in zip(real, gen):
            # Flatten in sequenza binaria (128*16)
            r_seq = r.flatten()
            g_seq = g.flatten()
            alignment = dtw(r_seq, g_seq, dist_method=lambda x, y: abs(x - y))
            dists.append(alignment.distance)  # .distance è la metrica finale

        return np.mean(dists)

    # ===================== MAIN EVALUATION ===================== #
    def evaluate(self, real_loader, num_batches=5, latent_dim=32):
        """
        real_loader: DataLoader di sequenze reali
        num_batches: numero di batch su cui calcolare le metriche
        """
        self.model.eval()
        pitch_scores, rhythm_scores, dtw_scores = [], [], []

        with torch.no_grad():
            for i, batch in enumerate(real_loader):
                if i >= num_batches:
                    break

                real = batch["real_melody_bar"].to(self.device)  # (B,1,128,16)

                # Genera samples random dal modello
                z = torch.randn(real.size(0), latent_dim).to(self.device)
                logits = self.model.decode(z)
                gen = self._binarize(logits)

                # Calcola metriche
                pitch_scores.append(self.pitch_distribution_similarity(real, gen))
                rhythm_scores.append(self.rhythm_similarity(real, gen))
                dtw_scores.append(self.dtw_distance(real, gen))

        return {
            "Pitch_Wass": float(np.mean(pitch_scores)),
            "Rhythm_Wass": float(np.mean(rhythm_scores)),
            "DTW": float(np.mean(dtw_scores)),
        }

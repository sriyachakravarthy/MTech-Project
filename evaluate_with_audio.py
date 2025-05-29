   
    def evaluate(self, data_loader):
        import time
        import psutil

        # --- Metric accumulators ---
        avg_sisnri = 0.0
        avg_sdri   = 0.0
        avg_pesqi  = 0.0
        avg_stoii  = 0.0

        # --- RTF accumulators ---
        total_audio_duration = 0.0  # in seconds
        total_forward_time = 0.0    # in seconds

        # --- Performance trackers ---
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.args.device)
        proc = psutil.Process()
        cpu_mem_before = proc.memory_info().rss / (1024 ** 2)  # MiB
        t_start = time.time()

        # load checkpoint & eval mode
        self._load_model(f'{self.args.checkpoint_dir}/last_best_checkpoint.pt')
        self.model.eval()

        with torch.no_grad():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()

            output_dir = os.path.join(self.args.checkpoint_dir, "eval_outputs")
            os.makedirs(output_dir, exist_ok=True)

            for i, (a_mix, a_tgt, ref_tgt) in enumerate(data_loader):
                if i >= 10:
                    break  # Limit to first 10 samples

                a_mix = a_mix.to(self.args.device)
                a_tgt = a_tgt.to(self.args.device)

                # Forward timing start
                fwd_start = time.time()
                a_tgt_est, _, _ = self.model(a_mix, ref_tgt)
                fwd_end = time.time()

                forward_time = fwd_end - fwd_start
                audio_duration = a_mix.shape[-1] / self.args.audio_sr

                total_forward_time += forward_time
                total_audio_duration += audio_duration
                print(f"[Sample {i+1}] Inference time: {forward_time:.4f} s | Duration: {audio_duration:.2f} s | RTF: {forward_time/audio_duration:.3f}")


                # SI-SNRi
                sisnri = (cal_SISNR(a_tgt, a_tgt_est) - cal_SISNR(a_tgt, a_mix)).item()
                avg_sisnri += sisnri

                # Convert to numpy
                est_np = a_tgt_est.squeeze().cpu().numpy()
                tgt_np = a_tgt.squeeze().cpu().numpy()
                mix_np = a_mix.squeeze().cpu().numpy()

                # SDRi
                sdri = SDR(tgt_np, est_np) - SDR(tgt_np, mix_np)
                avg_sdri += sdri

                # PESQi
                est_np /= np.max(np.abs(est_np)) + 1e-8
                pesqi = pesq(self.args.audio_sr, tgt_np, est_np, 'nb') - pesq(self.args.audio_sr, tgt_np, mix_np, 'nb')
                avg_pesqi += pesqi

                # STOIi
                stoii = stoi(tgt_np, est_np, self.args.audio_sr, extended=False) - stoi(tgt_np, mix_np, self.args.audio_sr, extended=False)
                avg_stoii += stoii

                # Save audio files (first 10 samples only)
                write_wav(os.path.join(output_dir, f"{i:02d}_mixture.wav"), self.args.audio_sr, mix_np)
                write_wav(os.path.join(output_dir, f"{i:02d}_target.wav"), self.args.audio_sr, tgt_np)
                write_wav(os.path.join(output_dir, f"{i:02d}_estimated.wav"), self.args.audio_sr, est_np)

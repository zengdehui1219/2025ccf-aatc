import os
import argparse

def create_scp(folder_paths, scp_path):
    with open(scp_path, 'w', encoding='utf-8') as f:
        for folder in folder_paths:
            for root, _, files in os.walk(folder):
                for file in files:
                    if file.endswith('.wav'):
                        full_path = os.path.abspath(os.path.join(root, file))
                        f.write(full_path + '\n')

def main():
    parser = argparse.ArgumentParser(description="Generate speech/noise/rir for training.")
    parser.add_argument("--speech_dir",default=['/data1/CCF2025/datasets_fullband/noisy_fullband/clean'])
    parser.add_argument("--noise_dir",default=['/data2/noise/ESC-50-master/audio','/data1/CCF2025/datasets_fullband/noise_fullband'])
    parser.add_argument("--rir_dir",default=['/data1/CCF2025/RIRS_NOISES/simulated_rirs','/data1/CCF2025/simulated_rirs_8k','/data1/CCF2025/simulated_rirs_16k'])

    args = parser.parse_args()
    
    create_scp(args.speech_dir,'./scp/speech.scp')
    create_scp(args.noise_dir,'./scp/noise.scp')
    create_scp(args.rir_dir,'./scp/rir.scp')

         

if __name__ == '__main__':
    main()   
"""Dataset class for audio spoofing detection."""

import torch
from torch.utils.data import Dataset


class AudioSpoofingDataset(Dataset):
    """
    Dataset class for audio spoofing detection.
    
    Inherits from PyTorch Dataset.
    """
    
    def __init__(self, data_path, metadata_path, split='train', transform=None, max_samples=None):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to raw or processed audio directory
            metadata_path: Path to protocol file
            split: Data split (train/val/test)
            transform: Optional transforms
            max_samples: Maximum number of samples to load (useful for fast CPU training)
        """
        import os
        self.data_path = data_path
        self.metadata_path = metadata_path
        self.split = split
        self.transform = transform
        self.max_samples = max_samples
        self.data = []
        self.labels = []
        self.speaker_ids = []
        self.system_ids = []
        
        self._load_metadata()
    
    def _load_metadata(self):
        """Load metadata from the protocol file."""
        import os
        import pandas as pd
        
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Protocol file not found: {self.metadata_path}")
            
        # Parse protocol file: SPEAKER_ID AUDIO_FILE_NAME - SYSTEM_ID KEY
        # E.g. LA_0079 LA_T_1138215 - - bonafide
        df = pd.read_csv(
            self.metadata_path, 
            sep=" ", 
            header=None, 
            names=["speaker_id", "audio_file", "col3", "system_id", "key"]
        )
        
        # Limit samples if requested
        if self.max_samples is not None:
            df = df.sample(n=min(self.max_samples, len(df)), random_state=42).reset_index(drop=True)
            
        for _, row in df.iterrows():
            filename = f"{row['audio_file']}.flac"
            filepath = os.path.join(self.data_path, filename)
            
            # Map key to binary label: bonafide -> 0, spoof -> 1
            label = 0 if row['key'] == 'bonafide' else 1
            
            # Save entries
            self.data.append(filepath)
            self.labels.append(label)
            self.speaker_ids.append(row['speaker_id'])
            self.system_ids.append(row['system_id'])
            
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.data)
    
    def __getitem__(self, idx: int):
        """Get item by index.
        
        Loads raw audio and returns the waveform tensor and class label.
        """
        import torch
        from src.preprocessing.audio_loader import load_audio
        
        filepath = self.data[idx]
        label = self.labels[idx]
        
        try:
            waveform, sr = load_audio(filepath, target_sr=16000, mono=True)
        except Exception as e:
            # Fallback to dummy silent waveform in case of read error during training
            waveform = torch.zeros(1, 64000)  # 4 seconds at 16kHz
            
        if self.transform:
            waveform = self.transform(waveform)
            
        return waveform, torch.tensor(label, dtype=torch.long)


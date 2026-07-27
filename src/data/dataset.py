"""Dataset class for audio spoofing detection."""

import logging
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Callable, Dict, List, Optional, Tuple, Union

from src.data.validator import validate_audio_file
from src.preprocessing.audio_loader import load_audio

logger = logging.getLogger(__name__)


class AudioSpoofingDataset(Dataset):
    """Dataset class for audio spoofing detection.
    
    Inherits from PyTorch Dataset. Supports lazy loading, path validation,
    corrupted file handling, and speaker filtering.
    """
    
    def __init__(
        self, 
        data_path: str, 
        metadata_path: str, 
        split: str = 'train', 
        transform: Optional[Callable] = None,
        validate_paths: bool = True,
        speakers: Optional[List[str]] = None
    ):
        """Initialize dataset.
        
        Args:
            data_path: Root directory of the raw/processed audio files.
            metadata_path: Path to the metadata CSV (manifest.csv).
            split: Data split name ('train', 'val'/'dev', 'test'/'eval').
            transform: Optional feature extractor or augmentation function.
            validate_paths: If True, checks file existence on initialization.
            speakers: Optional list of speaker IDs to filter the dataset.
        """
        self.data_path = data_path
        self.metadata_path = metadata_path
        self.split = split
        self.transform = transform
        self.validate_paths = validate_paths
        self.speakers = speakers
        
        self.data: List[Dict] = []
        self.labels: List[int] = []
        
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """Load metadata from CSV and filter by split and validation."""
        if not os.path.exists(self.metadata_path):
            logger.error(f"Metadata CSV file not found: {self.metadata_path}")
            raise FileNotFoundError(f"Metadata CSV file not found: {self.metadata_path}")
            
        logger.info(f"Loading metadata from {self.metadata_path}...")
        df = pd.read_csv(self.metadata_path)
        
        # Map requested split name to the split column values in CSV
        # ASVspoof2019 CSV split column has: 'train', 'dev', 'eval'
        split_mapping = {
            'train': 'train',
            'val': 'dev',
            'dev': 'dev',
            'test': 'eval',
            'eval': 'eval'
        }
        target_split = split_mapping.get(self.split.lower(), self.split)
        
        # Filter by split
        if 'split' in df.columns:
            df = df[df['split'].str.lower() == target_split.lower()]
            logger.info(f"Filtered split '{target_split}', found {len(df)} records.")
        else:
            logger.warning(f"Split column not found in metadata. Processing entire file list.")
            
        # Filter by speakers if provided
        if self.speakers is not None and 'speaker_id' in df.columns:
            df = df[df['speaker_id'].isin(self.speakers)]
            logger.info(f"Filtered by speakers, {len(df)} records remaining.")
            
        # Parse records
        valid_records = []
        label_mapping = {'real': 0, 'bonafide': 0, 'fake': 1, 'spoof': 1}
        
        # Path resolution logic
        for idx, row in df.iterrows():
            orig_path = row['filepath']
            
            # Normalize path slashes for the current OS
            norm_path = os.path.normpath(orig_path)
            
            # Resolve path: first check direct path, then check relative to data_path
            resolved_path = norm_path
            if not os.path.exists(resolved_path):
                resolved_path = os.path.join(self.data_path, norm_path)
                
            if not os.path.exists(resolved_path):
                # Try finding it relative to the parent workspace
                resolved_path = os.path.join(os.path.dirname(self.metadata_path), "..", norm_path)
                resolved_path = os.path.normpath(resolved_path)

            # If validate_paths is True, verify file existence
            if self.validate_paths and not os.path.exists(resolved_path):
                logger.debug(f"Skipping missing file: {orig_path} (resolved: {resolved_path})")
                continue
                
            label_str = str(row['label']).lower()
            label_id = label_mapping.get(label_str, 1)  # Default to spoof/fake if unrecognized
            
            record = {
                'filepath': resolved_path,
                'label': label_id,
                'attack_type': row.get('attack_type', '-'),
                'speaker_id': row.get('speaker_id', 'unknown'),
                'split': row.get('split', self.split)
            }
            valid_records.append(record)
            
        self.data = valid_records
        self.labels = [r['label'] for r in self.data]
        logger.info(f"Successfully loaded {len(self.data)} valid samples for split '{self.split}'.")
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get item by index.
        
        Lazy-loads audio, downmixes to mono, standardizes sample rate to 16 kHz.
        If loading fails due to corruption, attempts to fetch another sample.
        
        Args:
            idx: Index of the sample.
            
        Returns:
            Tuple of (waveform_or_feature_tensor, label).
        """
        # Ensure index is within range
        if idx < 0 or idx >= len(self.data):
            raise IndexError("Dataset index out of range.")
            
        record = self.data[idx]
        filepath = record['filepath']
        label = record['label']
        
        try:
            # Load and preprocess waveform (16 kHz, mono)
            waveform, sr = load_audio(filepath, target_sr=16000, mono=True)
            
            # Apply feature extraction transform if specified
            if self.transform is not None:
                features = self.transform(waveform)
                return features, label
                
            return waveform, label
            
        except Exception as e:
            logger.warning(f"Error loading audio file at index {idx} ({filepath}): {str(e)}. "
                           f"Attempting to load another sample.")
            # Fallback to a random valid index to prevent training loop crashes
            import random
            fallback_idx = random.randint(0, len(self.data) - 1)
            return self.__getitem__(fallback_idx)
            
    def get_metadata(self, idx: int) -> Dict:
        """Retrieve metadata for a specific sample.
        
        Args:
            idx: Index of the sample.
            
        Returns:
            Dictionary containing metadata of the sample.
        """
        if idx < 0 or idx >= len(self.data):
            raise IndexError("Dataset index out of range.")
        return self.data[idx]

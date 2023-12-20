import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
import math
from collections import OrderedDict
import torch.optim as optim
import h5py
import random
import datetime
import os
import torch.nn.functional as F
from face_alignment import FaceAlignment, LandmarksType
import torchaudio
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device {device}')
torch.cuda.empty_cache()

import neptune

run = neptune.init_run(
    project="oiegrant-personal/5527Project",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlNjNmYjcyNy04NzYxLTQzYzEtOWUwYy0zM2UxMzY2Y2I4ZTIifQ==",
)

def print_cuda_memory_usage():
    print(f"Current GPU memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Current GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, num_tokens):
        super().__init__()

        position = torch.arange(num_tokens).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(num_tokens, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class SpatioTemporalEncoder(nn.Module):
    def __init__(self, height, width, patch_size, frames, mlp_dim, d_model = 192, depth = 4, heads = 3, pool = 'cls', in_channels = 3, dropout = 0., emb_dropout = 0.):
        super().__init__()
        num_patches = (height // patch_size) * (width // patch_size)
        patch_dim = in_channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, d_model),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, frames, num_patches + 1, d_model))
        self.space_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.space_transformer = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True
        ), num_layers=depth)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True
        ), num_layers=depth)

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape

        cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b = b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        x += self.pos_embedding[:, :, :(n + 1)]
        x = self.dropout(x)

        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x)
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)

        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        x = self.temporal_transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        
        return x
    
class SpectroTemporalEncoder(nn.Module):
    def __init__(self, time_domain, freq_domain, depth, heads, kernel_size, mlp_dim, dropout):
        super(SpectroTemporalEncoder, self).__init__()
        self.encoder = nn.Sequential(OrderedDict([(f'SpectroTemporalEncoderLayer{i}',SpectroTemporalEncoderLayer(time_domain, freq_domain, heads, kernel_size, mlp_dim, dropout)) for i in range(depth)]))

    def forward(self, x):
        x = self.encoder(x)
        return x
    
class SpectroTemporalEncoderLayer(nn.Module):
    def __init__(self, time_domain, freq_domain, heads, kernel_size, mlp_dim, dropout):
        super(SpectroTemporalEncoderLayer, self).__init__()
        self.temporal_path = SpectroTemporalEncoderPath(time_domain, 2 * freq_domain, heads, kernel_size, mlp_dim, dropout)
        self.spectral_path = SpectroTemporalEncoderPath(2 * freq_domain, time_domain, heads, kernel_size, mlp_dim, dropout)

    def forward(self, x):
        x = self.temporal_path(x)
        y = x.transpose(1, 2)
        y = self.spectral_path(y)
        y = y.transpose(1, 2)
        x = x + y
        return x

class SpectroTemporalEncoderPath(nn.Module):
    def __init__(self, num_tokens, d_model, heads, kernel_size, mlp_dim, dropout):
        super(SpectroTemporalEncoderPath, self).__init__()
        self.pos_embedding = PositionalEncoding(d_model, num_tokens)
        self.disentangle = SourceDisentanglementCNN(num_tokens, kernel_size)
        self.encoderspectro = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True
        ), num_layers=1)

    def forward(self, x):
        x = self.pos_embedding(x)
        x = self.disentangle(x)
        x = self.encoderspectro(x)
        return x

class SourceDisentanglementCNN(nn.Module):
    def __init__(self, sequence_length, kernel_size):
        super(SourceDisentanglementCNN, self).__init__()
        self.convtemporal1 = nn.Conv1d(sequence_length, sequence_length, kernel_size=kernel_size, padding= 'same')
        self.convtemporal2 = nn.Conv1d(sequence_length, sequence_length, kernel_size=kernel_size, padding= 'same')
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.convtemporal1(x)
        x = self.relu(x)
        x = self.convtemporal2(x)
        return x
    
class FactorizedAttentionAVFusion(nn.Module):
    def __init__(self, visual_tokensize, acoustic_tokensize, num_acoustic_subspace):
        super(FactorizedAttentionAVFusion, self).__init__()
        self.mlpvisual = nn.Linear(visual_tokensize, visual_tokensize)
        self.mlpacoustic = nn.ModuleList([nn.Linear(acoustic_tokensize, acoustic_tokensize) for _ in range(num_acoustic_subspace)])
        self.softmax = nn.Softmax(dim = 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, visual_emb, acoustic_emb):
        acoustic_subspaces = torch.stack([FC(acoustic_emb) for FC in self.mlpacoustic], dim = 1)
        visual_emb = self.mlpvisual(visual_emb)
        visual_emb = self.softmax(visual_emb)
        x = torch.einsum('ij,ijkl->ikl', visual_emb, acoustic_subspaces)
        x = self.sigmoid(x)
        return x
    
class MLPComplexHead(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, freq_domain):
        super(MLPComplexHead, self).__init__()
        self.freq_domain = freq_domain
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = x.view(x.size(0), -1, self.freq_domain, 2)
        return x
    
class AVSpeechSeparator(nn.Module):
    def __init__(self, audio_samples, frames, image_height, image_width, patch_size, d_model, depth, heads, mlp_dim, dropout, n_fft, hop_length, win_length):
        super(AVSpeechSeparator, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.time_domain = 1 + audio_samples // hop_length
        self.freq_domain = n_fft // 2 + 1
        self.audio_samples = audio_samples
        self.visual_encoder = SpatioTemporalEncoder(image_height, image_width, patch_size, frames, mlp_dim, d_model, depth, heads) 
        self.audio_encoder = SpectroTemporalEncoder(self.time_domain, self.freq_domain, depth, heads, patch_size, mlp_dim, dropout)
        self.modality_fusion = FactorizedAttentionAVFusion(d_model, self.freq_domain * 2, d_model)
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.freq_domain * 2,
                nhead=heads,
                dim_feedforward=mlp_dim,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=depth)
        complex_head_dim = complex_head_dim = self.time_domain * self.freq_domain * 2 + self.freq_domain * 2
        self.mask_generation_network = MLPComplexHead(complex_head_dim, mlp_dim, complex_head_dim - (2 * self.freq_domain), self.freq_domain)

    def complex_product(self, spectrogram, mask):
        real = spectrogram[..., 0] * mask[..., 0] - spectrogram[..., 1] * mask[..., 1]
        imag = spectrogram[..., 0] * mask[..., 1] + spectrogram[..., 1] * mask[..., 0]
        return torch.stack([real, imag], dim=-1)

    def forward(self, visual_stream, acoustical_stream):
        visual_emb = self.visual_encoder(visual_stream)
        spectrogram = torch.stft(acoustical_stream, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, return_complex=False)
        flattened_spectrogram = spectrogram.transpose(1, 2) # p.u. stinky stinky stinky
        flattened_spectrogram = torch.flatten(flattened_spectrogram, start_dim = 2)
        acoustical_emb = self.audio_encoder(flattened_spectrogram)
        #av_emb = self.modality_fusion(visual_emb, acoustical_emb)
        visual_emb = visual_emb.unsqueeze(1)
        print(f'visual emb {visual_emb.shape}')
        print(f'acoustic emb {acoustical_emb.shape}')
        av_emb = torch.cat((visual_emb, acoustical_emb), dim = 1)
        decoded_emb = self.decoder(av_emb)
        complex_mask = self.mask_generation_network(decoded_emb)
        complex_mask = complex_mask.transpose(1, 2) # p.u. stinky stinky stinky
        separated_spectrogram = self.complex_product(spectrogram, complex_mask)
        separated_spectrogram = torch.view_as_complex(separated_spectrogram)
        separated_audio = torch.istft(separated_spectrogram, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, length = self.audio_samples, return_complex=False)
        return separated_audio
    
class SI_SDR(nn.Module):  #Zero-mean
    def __init__(self):
        super(SI_SDR, self).__init__()

    def forward(self, target, estimate):
        signal =  (torch.einsum('ij,ij->j', target, estimate) * target) / (torch.norm(target, dim=1) ** 2).unsqueeze(1)
        distortion = estimate - signal
        loss = -10 * torch.log10((torch.norm(signal, dim=1) ** 2) / (torch.norm(distortion, dim=1) ** 2))
        loss = torch.sum(loss)/target.shape[0]
        return loss

class HDF5:
    def __init__(self, path):
        self.path = path

    def write(self, sample_id, video_batch, audio_batch, frames, rgb_channels, image_size, audio_samples):
        with h5py.File(self.path, 'a') as hf:
            if 'video_tensor' not in hf:
                hf.create_dataset('video_tensor', shape=(0, frames, rgb_channels, image_size, image_size), maxshape=(None, frames, rgb_channels, image_size, image_size), dtype='float32')

            if 'audio_tensor' not in hf:
                hf.create_dataset('audio_tensor', shape=(0, audio_samples), maxshape=(None, audio_samples), dtype='float32')

            hf['video_tensor'].resize((sample_id + 1, frames, rgb_channels, image_size, image_size))
            hf['video_tensor'][sample_id] = video_batch

            hf['audio_tensor'].resize((sample_id + 1, audio_samples))
            hf['audio_tensor'][sample_id] = audio_batch

    def read_video_tensor(self, i):
        with h5py.File(self.path, 'r') as hf:
            for key in hf.keys():
                if key.startswith('video_tensor'):
                    video_tensor = torch.from_numpy(hf[key][i])
        return video_tensor

    def read_audio_tensor(self, i):
        with h5py.File(self.path, 'r') as hf:
            for key in hf.keys():
                if key.startswith('audio_tensor'):
                    audio_tensor = torch.from_numpy(hf[key][i])
        return audio_tensor

    def read_tensors(self, i):
        video_tensor = self.read_video_tensor(i)
        audio_tensor = self.read_audio_tensor(i)
        return video_tensor, audio_tensor

def train_model(model, criterion, optimizer, scheduler, visual_stream, audio_stream, target):
    optimizer.zero_grad()
    estimate = model(visual_stream, audio_stream)
    loss = criterion(target, estimate)
    loss.backward()
    optimizer.step()
    scheduler.step(loss.item())
    return loss


def get_audio_agumentation_index(start, end, num_samples):
    i = -1
    while i < 0:
        random_number = random.randint(0, num_samples - 1)
        if random_number < start or random_number > end:
            i = random_number
    return i

def save_loss(path, loss):
  with open(path, 'a') as file:
      file.write(str(loss) + '\n')
        
def save_samples(model, num_samples, total_samples, batch, epoch):
    random_indices = [random.randint(0, total_samples) for _ in range(num_samples)]
    gain = 0.1
    for index in random_indices:
        video_sample, audio_sample = hdf5_file.read_tensors(index)
        audio_sample = rms_normalization(audio_sample, gain)
        audio_augmentation = hdf5_file.read_audio_tensor(random.randint(0, total_samples))
        audio_augmentation = rms_normalization(audio_augmentation, gain)
        noisy_audio = audio_sample + audio_augmentation
        cleaned_audio = model(video_sample.unsqueeze(dim=0).to(device), noisy_audio.unsqueeze(dim=0).to(device))
        write_sample(noisy_audio, audio_sample, cleaned_audio.squeeze(dim=0).to('cpu').detach(), batch, index, epoch)

def write_sample(noisy_audio_tensor, clean_audio_tensor, cleaned_audio_tensor, batch, sample, epoch):
  root = "sample_data_no_disentangle_L1Loss_Scheduler_500steps_2SpeakerRMSNormal"
  epoch_dir = f'epoch{epoch}'
  batch_dir = f'batch{batch}'
  sample_dir = f'sample{sample}'
  path = os.path.join(root, epoch_dir, batch_dir, sample_dir)
  if not os.path.exists(path):
    os.makedirs(path)

  torchaudio.save(os.path.join(path, "noisy_audio.wav"), noisy_audio_tensor.unsqueeze(dim=0), 44100)
  torchaudio.save(os.path.join(path, "clean_audio.wav"), clean_audio_tensor.unsqueeze(dim=0), 44100)
  torchaudio.save(os.path.join(path, "cleaned_audio.wav"), cleaned_audio_tensor.unsqueeze(dim=0), 44100)

def rms_normalization(audio_tensor, gain):
    rms = torch.sqrt(torch.mean(audio_tensor ** 2))
    normal_audio_tensor = (audio_tensor/ rms) * gain
    return normal_audio_tensor

frames = 75
batch_size = 2
rgb_channels = 3
image_size = 200
height = 32
width = 64
num_audio_samples = 132300
patch_size = 4
d_model = 402
heads = 6
depth = 12
mlp_dim = 1024
dropout = 0.1

n_fft = 400
hop_length = 197
win_length = 400

gain = 0.1


# csv_file_path = 'avspeech_train.csv'
# column_names = ['YouTube URL', 'Segment Start Time', 'Segment End Time', 'Face X Coordinate', 'Face Y Coordinate']
# df = pd.read_csv(csv_file_path, header = None, names = column_names)

#clip_duration = 3
#crop_size = image_size
#fps = 25
#res = '360p'
#av_file_set = AVFileSet(df, clip_duration, res, crop_size, fps)
#batch_video_list = []
#batch_audio_list = []
#previous_batch_audio_tensor = None

lr = 3e-5

# neptune params
params = {
    "lr":lr,
    "dmodel":d_model,
    "mlp_dim":mlp_dim,
    "dropout":dropout,
    "depth":depth,
    "heads":heads,
    "batch_size":batch_size,
}
run["parameters"] = params


model = AVSpeechSeparator(num_audio_samples, frames, height, width, patch_size, d_model, depth, heads, mlp_dim, dropout, n_fft, hop_length, win_length)
model.to(device)
print_cuda_memory_usage()

#criterion = SI_SDR()
#criterion = nn.MSELoss()
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr = lr)
scheduler = ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.5, patience = 500, verbose = True)


hdf5_filename = '../Dataset/train_data_high_res.h5'
hdf5_file = HDF5(hdf5_filename)
loss_filename = 'loss_high_res_no_disentangle_L1Loss_Scheduler_2SpeakerRMSNormal.txt'
model_weights_path = 'model_weights_high_res_no_disentangle_L1Loss_Scheduler_2SpeakerRMSNormal.pth'

num_samples = 2500
epochs = 10000

# num_samples = 8
# epochs = 2

k = 0
model.eval()
save_samples(model, 4, num_samples, k, 0)
model.train()

for i in range(epochs):
    for j in range(num_samples // batch_size):
        if j % 1000 == 0:
            k += 1
            model.eval()
            save_samples(model, 4, num_samples, k * 1000, i)
            model.train()
        start_time = datetime.datetime.now()
        print(f"Epoch: {i}, Batch: {j} ")
        start = j * batch_size
        end = start + batch_size
        video_samples = []
        audio_samples = []
        audio_augmentations = []
        print('Loading batch...')
        for k in range(start, end):
            video_sample, audio_sample = hdf5_file.read_tensors(k)
            audio_sample = rms_normalization(audio_sample, gain)
            #video_sample = video_sample[::3, :, :, :]
            audio_augmentation_index = get_audio_agumentation_index(start, end, num_samples)
            audio_augmentation = hdf5_file.read_audio_tensor(audio_augmentation_index)
            audio_augmentation = rms_normalization(audio_augmentation, gain)
            #audio_augmentation = 0.1 * torch.rand(num_audio_samples)
            video_samples.append(video_sample)
            audio_samples.append(audio_sample)
            audio_augmentations.append(audio_augmentation)
        video_samples = torch.stack(video_samples)
        audio_samples = torch.stack(audio_samples)
        audio_augmentations = torch.stack(audio_augmentations)
        noisy_audio_samples = audio_samples + audio_augmentations
        video_samples = video_samples.to(device)
        print_cuda_memory_usage()
        audio_samples = audio_samples.to(device)
        print_cuda_memory_usage()
        noisy_audio_samples = noisy_audio_samples.to(device)
        print_cuda_memory_usage()
        print('Done!')
        loss = train_model(model, criterion, optimizer, scheduler, video_samples, noisy_audio_samples, audio_samples)
        save_loss(loss_filename, loss)
        print(loss)
        
        end_time = datetime.datetime.now()
        diff_time = end_time - start_time
        print(f"Iteration {i + 1}: Time - {diff_time}")

        #neptune loss capture
        run["train/batch/loss"].append(loss)
        
        if j % 1000 == 0:
            if os.path.exists(model_weights_path):
              os.remove(model_weights_path)
            torch.save(model.state_dict(), model_weights_path)
from sp_styletts2 import *
import scipy.io.wavfile
# import gradio as gr
from app2 import *
from try_noise import *
import demucs.separate
import time



passage = '''बारामतीत सुनेत्रा पवार विरुद्ध सुप्रिया सुळे अशी लोकसभा निवडणुकीतली लढत पाहण्यास मिळेल या चर्चा लोकसभा निवडणुकीच्या आधी रंगल्या आहेत. सुप्रिया सुळे या आता कशी तयारी करणार? तसंच बारामती लोकसभा मतदारसंघात कुणाचं पारडं जड ठरणार? या चर्चा रंगल्या आहेत. अशातच सुप्रिया सुळे यांनी पवार कुटुंबाबाबत महत्त्वाचं वक्तव्य केलं आहे. तसंच अमित शाह यांनाही त्यांनी उत्तर दिलं आहे.'''

s=time.time()
sentences = passage.split('.') # simple split by comma
wavs = []
s_prev = None
for text in sentences:
    if text.strip() == "": continue
    text += '.' # add it back
    noise = torch.randn(1,1,256).to(device)
    wav, s_prev = LFinference(text, s_prev, noise, alpha=0.7, diffusion_steps=10, embedding_scale=1.5)
    wavs.append(wav)

scipy.io.wavfile.write("output.wav",24000, np.concatenate(wavs))

info, (tgt_sr, audio_opt) = tts("output.wav")
scipy.io.wavfile.write("output_rvc.wav",tgt_sr, audio_opt)
print(info)

# separate()


demucs.separate.main(["--mp3", "--two-stems", "vocals", "-n", "mdx_extra", "output_rvc.wav"])
print(time.time()-s)




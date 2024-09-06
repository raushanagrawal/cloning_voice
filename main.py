from sp_styletts2 import *
import scipy.io.wavfile
# import gradio as gr
import uvicorn
from app2 import *
from try_noise import *
import demucs.separate
import time
from fastapi import APIRouter,File, UploadFile,Query,Depends,HTTPException,FastAPI,Body
from fastapi.responses import FileResponse, PlainTextResponse,JSONResponse
import os
import logging
import tempfile
import uuid
from audiostretchy.stretch import stretch_audio
from Log.log import create_logger
app = FastAPI()

#Directory paths
input_scripts = "input_scripts"
generated_audios = "generated_audios"

input_file = "input_file"
output_file = "output_audios"


#Ensure directories exist
# os.makedirs(input_scripts,exist_ok=True)
# os.makedirs(generated_audios, exist_ok=True)
os.makedirs(input_file,exist_ok=True)
os.makedirs(output_file, exist_ok=True)

logger1 = create_logger("Log/log.log")

# @app.post('/generate_audio')
# async def upload_text_file(file:UploadFile = File(...)):
#     file_content = (await file.read()).decode("utf-8")
#     # passage = '''बारामतीत सुनेत्रा पवार विरुद्ध सुप्रिया सुळे अशी लोकसभा निवडणुकीतली लढत पाहण्यास मिळेल या चर्चा लोकसभा निवडणुकीच्या आधी रंगल्या आहेत. सुप्रिया सुळे या आता कशी तयारी करणार? तसंच बारामती लोकसभा मतदारसंघात कुणाचं पारडं जड ठरणार? या चर्चा रंगल्या आहेत. अशातच सुप्रिया सुळे यांनी पवार कुटुंबाबाबत महत्त्वाचं वक्तव्य केलं आहे. तसंच अमित शाह यांनाही त्यांनी उत्तर दिलं आहे.'''

#     wavs = []
#     s=time.time()
#     # passage=file_content.split('\n')
#     # for sentences in passage:
#     sentences = file_content.split('.') # simple split by comma
#     print('Sentences: ',sentences)
#     # sentences = passage.split('.')
    
#     s_prev = None
#     for text in sentences:
#         print('Text: ',text)
#         if text.strip() == "":
#             continue
#         text += '.' # add it back
#         noise = torch.randn(1,1,256).to(device)
#         wav, s_prev = LFinference(text, s_prev, noise, alpha=0.7, diffusion_steps=10, embedding_scale=1.5)
#         wavs.append(wav)
        
#     unique_id = str(uuid.uuid1())[:4]
#     original_file = file.filename
#     print('original file: ',original_file)

#     input_original_file = f"{original_file.split('.')[0]}_{unique_id}.txt"
#     input_original_path = os.path.join(input_scripts,input_original_file)
#     with open(input_original_path,"w") as input_script_file:
#         input_script_file.write(file_content)


#     output_wav_file = f"{original_file.split('.')[0]}_{unique_id}.wav"
#     scipy.io.wavfile.write(output_wav_file,24000, np.concatenate(wavs))

#     info, (tgt_sr, audio_opt)= tts(output_wav_file)
#     output_rvc_file = f"{original_file.split('.')[0]}_{unique_id}_rvc.wav"
#     scipy.io.wavfile.write(output_rvc_file,tgt_sr, audio_opt)
#     print(info)
#     return FileResponse(output_rvc_file, media_type="audio/wav", filename=output_rvc_file)


@app.post('/upload_line_by_line')
async def upload_text_file_line(file:UploadFile = File(...)):
    if not file.filename.endswith(".txt"):
        logger1.info(f'Only text files(.txt) are allowed')
        raise HTTPException(
            status_code = 400,
            detail = "Only text files(.txt) are allowed"
        )

    #generate a UUID
    unique_id = str(uuid.uuid1())[:2]
    file_path = os.path.join(input_file,f"{file.filename.split('.')[0]}_{unique_id}.txt")

    #Save the uploaded file
    with open(file_path,"wb") as buffer:
        buffer.write(await file.read())

    logger1.info(f'Saving uploaded file:',f"{file.filename.split('.')[0]}_{unique_id}.txt")

    #Create output file before
    output_file_path = os.path.join(output_file,f"{os.path.splitext(file.filename)[0]}_{unique_id}.wav")
    logger1.info(f'Creating output file')

    wavs = []
    s=time.time()
    with open(file_path,'r',encoding="utf-8") as text_file:
        logger1.info(f'Opening Text file in read mode.')
        content = text_file.read()
        sentences = content.split('.')           #simple split by '.'
        print('Sentences: ',sentences)
        logger1.info(f'Sentences:{sentences}')
        s_prev = None
        for text in sentences:
            text = text.strip()
            print('Text before: ',text)
            logger1.info(f'Text:{text}')
            text += '.'                  
            try:
                noise = torch.randn(1,1,256).to(device)
                wav, s_prev = LFinference(text, s_prev, noise, alpha=0.7, diffusion_steps=10, embedding_scale=1.5) #send tokenized text to style tts
                wavs.append(wav)
            except Exception as e:
                # print(f"Error processing text: {text}. ({e})")
                logger1.info(f"Error processing text: {text}. ({e})")
                subtexts = text.split(',')                        
                print('subtexts: ',subtexts)
                for subtext in subtexts:
                    print('subtext: ',subtext)
                    logger1.info(f'Subtext:{subtext}')
                    noise = torch.randn(1,1,256).to(device)
                    wav, s_prev = LFinference(subtext, s_prev, noise, alpha=0.7, diffusion_steps=10, embedding_scale=1.5)
                    wavs.append(wav)
                continue
    e = time.time()
    print('Time Taken for StyleTTS to process: ',e-s)
    logger1.info(f'Time Taken for StyleTTS to process:{e-s}')

    scipy.io.wavfile.write(output_file_path,24000, np.concatenate(wavs))
    filename=output_file_path.split('/')[1].split('.')[0]
    demucs.separate.main(["--mp3", "--two-stems", "vocals", "-n", "mdx_extra", output_file_path])
    print(os.path.exists(f'/home/infogen-1/Music/Audio_SP/Audio_SP_cloning/separated/mdx_extra/{filename}/vocals.mp3'),f'/home/infogen-1/Music/Audio_SP/Audio_SP_cloning/separated/mdx_extra/{filename}/vocals.mp3')


    info, (tgt_sr, audio_opt)= tts(f'/home/infogen-1/Music/Audio_SP/Audio_SP_cloning/separated/mdx_extra/{filename}/vocals.mp3')               #Pass output of style tts to RVC model
    #audio_static_file = '/home/infogen-1/Downloads/1_newmodel80_(Vocals).wav'
    #info, (tgt_sr, audio_opt)= tts(audio_static_path)      
    ee = time.time()
    print('Time Taken for RVC model to process:  ',ee-e)
    logger1.info(f'Time Taken for RVC model to process: {ee-e}')
    scipy.io.wavfile.write(output_file_path,tgt_sr, audio_opt)

    #stretch_audio(output_file_path,output_file_path,ratio=1.2)     #Adjust audio speed
    logger1.info(f'Adjusted Audio Speed of file {os.path.splitext(file.filename)[0]}_{unique_id}.wav')

    print(info)
    return FileResponse(output_file_path, media_type="audio/wav", filename=os.path.basename(output_file_path))




if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)

    # separate()


    # demucs.separate.main(["--mp3", "--two-stems", "vocals", "-n", "mdx_extra", "output_rvc.wav"])
    # print(time.time()-s)




#passage = '''बारामतीत सुनेत्रा पवार विरुद्ध सुप्रिया सुळे अशी लोकसभा निवडणुकीतली लढत पाहण्यास मिळेल या चर्चा लोकसभा निवडणुकीच्या आधी रंगल्या आहेत. सुप्रिया सुळे या आता कशी तयारी करणार? तसंच बारामती लोकसभा मतदारसंघात कुणाचं पारडं जड ठरणार? या चर्चा रंगल्या आहेत. अशातच सुप्रिया सुळे यांनी पवार कुटुंबाबाबत महत्त्वाचं वक्तव्य केलं आहे. तसंच अमित शाह यांनाही त्यांनी उत्तर दिलं आहे.'''

# s=time.time()
# sentences = passage.split('.') # simple split by comma
# wavs = []
# s_prev = None
# for text in sentences:
#     if text.strip() == "": continue
#     text += '.' # add it back
#     noise = torch.randn(1,1,256).to(device)
#     wav, s_prev = LFinference(text, s_prev, noise, alpha=0.7, diffusion_steps=10, embedding_scale=1.5)
#     wavs.append(wav)

# scipy.io.wavfile.write("output.wav",24000, np.concatenate(wavs))

# info, (tgt_sr, audio_opt) = tts("output.wav")
# scipy.io.wavfile.write("output_rvc.wav",tgt_sr, audio_opt)
# print(info)

# # separate()


# demucs.separate.main(["--mp3", "--two-stems", "vocals", "-n", "mdx_extra", "output_rvc.wav"])
# print(time.time()-s)



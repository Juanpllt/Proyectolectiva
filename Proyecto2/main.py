import librosa
import numpy as np
import matplotlib.pyplot as graficoFrecuencia 
import soundfile as guardoAudio
from scipy.fft import fft, fftfreq


rutaAudio = 'Proyecto 2/audio2.wav'
senial, sr = librosa.load(rutaAudio, sr=None)

intervalosSilenciados = librosa.effects.split(senial, top_db=20)
senialFiltrada = np.concatenate([senial[start:end] for start, end in intervalosSilenciados])


guardoAudio.write('Proyecto 2/audioFiltrado.wav', senialFiltrada, sr)

def quantize(senial, levels):
    
    senialNormal = senial / np.max(np.abs(senial))
    senialCuantificada = np.round((senialNormal + 1) * (levels / 2 - 1)).astype(int)
    return senialCuantificada

senialCuantificada = quantize(senialFiltrada, 256)  

yFrecuencia = fft(senialFiltrada)
num = len(senialFiltrada)
xFrecuencia = fftfreq(num, 1 / sr)


graficoFrecuencia.figure(figsize=(10, 6))
graficoFrecuencia.plot(xFrecuencia[:num // 2], np.abs(yFrecuencia[:num // 2])) 
graficoFrecuencia.title('Grafico del dominio de frecuencia')
graficoFrecuencia.xlabel('Frecuencia (Hz)')
graficoFrecuencia.ylabel('amplitud')
graficoFrecuencia.grid()
graficoFrecuencia.show()


frecuenciaDominante = xFrecuencia[np.argmax(np.abs(yFrecuencia[:num // 2]))]
print(f"La frecuencia dominante es  {frecuenciaDominante}Hz")



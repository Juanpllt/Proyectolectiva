import cv2 as disenoImagen, matplotlib.pyplot as resultado


rutaImagen = 'C:/Users/juanp/Escritorio/ProyectosElectiva/Proyecto 1/imagenes/objetos.jpg' 
imagen = disenoImagen.imread(rutaImagen)
color= disenoImagen.cvtColor(imagen, disenoImagen.COLOR_BGR2Lab )

#Propiedades del contorno de la imagen

ruido = disenoImagen.medianBlur(color, 5)
esquinas = disenoImagen.Canny(ruido, 60, 200)
dilatado = disenoImagen.dilate(esquinas, None, iterations=20)
contorno , _ = disenoImagen.findContours(dilatado, disenoImagen.RETR_EXTERNAL, disenoImagen.CHAIN_APPROX_SIMPLE)


area = 900
#Resultado Contorno figura
contornoFigura = [con for con in contorno if disenoImagen.contourArea(con) > area]


#Dibujando contorno de los objetos en la imagen
disenoImagen.drawContours(imagen, contornoFigura, -20, (0, 0, 255), 10)


resultado.figure(figsize=(7, 7))
resultado.imshow(disenoImagen.cvtColor(imagen, disenoImagen.COLOR_BGR2RGB))
resultado.title('Figuras encontradas')
resultado.axis('on')
resultado.show()

#Numero de objetos encotrados
print(f'Cantidad de figuras encontrada: {len(contornoFigura)}')
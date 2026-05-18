from PIL import Image

# Load your PNG
img = Image.open("C:\\Users\\Sraglow\\Documents\\GitHub\\PL_Analysis\\tauc.png")

# Convert and save as ICO (multiple sizes is best for Windows)
img.save(
    "C:\\Users\\Sraglow\\Documents\\GitHub\\PL_Analysis\\tauc.ico",
    format="ICO",
    sizes=[(16,16), (32,32), (48,48), (64,64), (128,128), (256,256)]
)
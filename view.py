import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

directory = "./output"
files = sorted([f for f in os.listdir(directory) if f.endswith('.jpg')])
index = 0 

fig, ax = plt.subplots()

def update_display():
    ax.clear()
    
    image_path = os.path.join(directory, files[index])
    txt_path = image_path.replace('.jpg', '.txt')

    img = mpimg.imread(image_path)
    ax.imshow(img)
    ax.axis('off')

    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            fen = f.readline().strip()
    else:
        fen = "FEN not found."

    ax.set_title(f"FEN: {fen}", fontsize=10)
    fig.canvas.draw()

def on_key(event):
    global index
    if event.key == 'right':
        index = (index + 1) % len(files)
        update_display()
    elif event.key == 'left':
        index = (index - 1) % len(files)
        update_display()
    elif event.key == 'escape':
        plt.close(fig)

print("Use the arrow keys to navigate images, press escape to close program")
fig.canvas.mpl_connect('key_press_event', on_key)
update_display()
plt.show()

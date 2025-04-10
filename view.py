import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

output_directory = "./output"
image_directory = "./datasets/chess/test/images/"
files = sorted([f for f in os.listdir(output_directory) if f.endswith('.jpg')])
index = 0 

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))  

def update_display():
    ax1.clear()
    ax2.clear()

    output_image_path = os.path.join(output_directory, files[index])
    original_filename = files[index].replace("result_", "")
    original_image_path = os.path.join(image_directory, original_filename)
    txt_path = output_image_path.replace('.jpg', '.txt')

    output_img = mpimg.imread(output_image_path)
    ax2.imshow(output_img)
    ax2.set_title("Output Image")
    ax2.axis('off')

    if os.path.exists(original_image_path):
        original_img = mpimg.imread(original_image_path)
        ax1.imshow(original_img)
        ax1.set_title("Original Image")
    else:
        ax1.text(0.5, 0.5, 'Original image not found.', ha='center', va='center')
    ax1.axis('off')

    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            fen = f.readline().strip()
    else:
        fen = "FEN not found."

    fig.suptitle(f"FEN: {fen}", fontsize=10)
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

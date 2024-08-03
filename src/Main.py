# * Author : NTI Post Graduate Diploma 2023/2024  (Arabic Recognition Project)
import tkinter as tk
from turtle import width
from numpy import pad
from tkthread import tk, TkThread
from tkinter import *
from tkinter import filedialog, PhotoImage
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
from OCR import main
import shutil
import os
from tkinter.ttk import *


font_family = "Raavi"
text_font = (font_family, 18)
label_font = (font_family, 11)
bg_color = "aliceblue"
btn_bg_color = "#c1d0da"
btn_width = 14
fontDict = {
    "Calibri": 3,
    "Arial": 3,
    "Dubai Medium": 3,
    "Courier New": 6,
    "Times New Roman": 3,
    "Tahoma": 3,
    "Microsoft Sans Serif": 3,
    "Segoe UI": 4,
}

root = tk.Tk()

text_widget = tk.Text(root, wrap="word", width=100, height=15)
cut_var = tk.StringVar()
save_img_var = tk.IntVar()
timer_id = None
image_label = None

ar_lang = "Arabic"
en_lang = "English"


# Text dictionaries for English and Arabic
align_dict = {en_lang: (0, 1, 2, 3, 4, 5), ar_lang: (5, 3, 2, 1, 0)}
text_dict = {
    en_lang: {
        "title": "Arabic OCR",
        "import_file": "Choose Photo",
        "select_font": "Select the Font",
        "word_space": "Words Space",
        "save_img": "Save Image Segments",
        "clear": "Clear",
        "submit": "Submit",
        "language": "Language:",
        "ocr_error_less": "Word Space should not be less than 2",
        "ocr_error_more": "Word Space should not be greater than 10",
    },
    ar_lang: {
        "title": "OCR العربية",
        "import_file": "اختر صورة",
        "select_font": "اختر الخط",
        "word_space": "المسافة بين الكلمات",
        "save_img": "احفظ الصور المقطعة",
        "clear": "مسح",
        "submit": "تحويل الصورة",
        "language": "لغة",
        "ocr_error_less": "مسافة الكلمة يجب أن لا تقل عن 2",
        "ocr_error_more": "مسافة الكلمة يجب أن لا تزيد عن 10",
    },
}

current_language = "English"


# Function to recreate directories
def recreateDirectory(folders):
    for folder in folders :
        if os.path.exists(folder):
            shutil.rmtree(folder, ignore_errors=False, onerror=None)
        os.mkdir(folder)


fileName = ""


# Function to import file
def import_file():
    global image_label, file_path, fileName
    file_path = filedialog.askopenfilename(
        title=text_dict[current_language]["import_file"],
        filetypes=[
            ("png", "*.png"),
            ("bmp", "*.bmp"),
            ("jpg", "*.jpg"),
            ("jpeg", "*.jpeg"),
        ],
    )
    if file_path:
        # Process the selected file (you can replace this with your own logic)
        fileName = file_path.split("/").pop()
        recreateDirectory(["test","lines","words","chars"])
        text_widget.delete(1.0, tk.END)
        dest_path = f"test/{fileName}"
        if dest_path.find(".jpeg") != -1:
            dest_path = dest_path.replace(".jpeg", ".png")

        print("Selected file:", file_path, "dest_path:", f"{dest_path}")

        shutil.copyfile(file_path, dest_path)

        # Display selected image
        img = Image.open(file_path)

        # Calculate the target size while maintaining aspect ratio
        target_width = 1100
        target_height = 350
        img_aspect_ratio = img.width / img.height
        target_aspect_ratio = target_width / target_height

        if img_aspect_ratio > target_aspect_ratio:
            # if img_aspect_ratio > target_aspect_ratio:
            new_width = target_width
            new_height = int(target_width / img_aspect_ratio)
        else:
            new_height = target_height
            new_width = int(target_height * img_aspect_ratio)

        img = img.resize((new_width, new_height), Image.LANCZOS)
        img = ImageTk.PhotoImage(img)

        if image_label is None:
            image_label = Label(root, image=img)
            image_label.image = img
            image_label.grid(
                row=4, rowspan=2, column=0, columnspan=5, padx=10, pady=10
            )  # Updated

        else:
            image_label.configure(image=img)
            image_label.image = img
            image_label.grid(
                row=4, rowspan=2, column=0, columnspan=5, padx=10, pady=10
            )  # Updated


# Function to start OCR process
def start_ocr():
    save_imgs = save_img_var.get()
    print("save_imgs:", f"{save_imgs}")
    cut_val = cut_var.get()
    cut = 3
    if cut_val.isnumeric() and cut_val != "":
        cut = float(cut_val)
        if cut < 2:
            messagebox.showerror(
                "OCR Error", text_dict[current_language]["ocr_error_less"]
            )
            return
        if cut > 10:
            messagebox.showerror(
                "OCR Error", text_dict[current_language]["ocr_error_more"]
            )


    main(cut, save_imgs)
    text_name = f'output/text/{fileName.split(".")[0]}.txt'
    print("text_name:", f"{text_name}")
    if os.path.exists(text_name):
        with open(text_name, "r", encoding="utf8") as file:
            content = file.read()
            text_widget.delete(1.0, tk.END)  # Clear previous content
            text_widget.insert(tk.END, content, "tag-right")


# Function to clear and reset all elements in the GUI
def clear_all():
    recreateDirectory(["test","lines","words","chars"])
    global image_label, file_path
    file_path = None
    cut_var.set("")
    if image_label:
        image_label.destroy()
        image_label = None
    text_widget.delete(1.0, tk.END)
    fontChoosen.set("")


# Function to create combobox
def comboBoxCreation():
    # Combobox creation
    n = tk.StringVar()
    # Label
    select_font_label = ttk.Label(
        root,
        text=text_dict[current_language]["select_font"],
        font=label_font,
        background=bg_color,
    )
    fontChoosen = ttk.Combobox(root, width=27, font=label_font, textvariable=n)
    fontChoosen.bind("<<ComboboxSelected>>", on_select)
    # Adding combobox drop down list
    fontChoosen["values"] = list(fontDict.keys())
    fontChoosen.grid(column=1, row=1)
    select_font_label.grid(column=0, row=1, padx=10)
    fontChoosen.current(0)
    return select_font_label, fontChoosen


# Function to handle combobox selection
def on_select(event):
    selected = event.widget.get()
    cut_var.set(fontDict[selected])


# Function to change language
def change_language(selected_language):
    global current_language
    current_language = selected_language
    update_language()


# Function to update text and styles based on the selected language
def update_language():
    root.title(text_dict[current_language]["title"])
    import_button.config(text=text_dict[current_language]["import_file"])
    cut_label.config(text=text_dict[current_language]["word_space"])
    clear_button.config(text=text_dict[current_language]["clear"])
    submit_button.config(text=text_dict[current_language]["submit"])
    # language_label.config(text=text_dict[current_language]["language"])
    select_font_label.config(text=text_dict[current_language]["select_font"])
    save_img_label.config(text=text_dict[current_language]["save_img"])
    align_widget()


def align_widget():
    align = align_dict[current_language]
    select_font_label.grid(row=1, column=align[0], padx=10)
    fontChoosen.grid(row=1, column=align[1])
    import_button.grid(row=1, column=align[2], padx=10, pady=10)
    clear_button.grid(row=1, column=align[3], padx=10, pady=10)
    submit_button.grid(row=1, column=align[4], padx=10, pady=10)

    cut_label.grid(row=3, column=align[0], padx=10, pady=10)
    cut_entry.grid(row=3, column=align[1], padx=10, pady=10)
    save_img_label.grid(row=3, column=align[2], pady=10)
    save_img_check.grid(row=3, column=align[3], pady=10)

    # Update other widget texts and styles if necessary


# Main function
if __name__ == "__main__":
    # root.geometry("1200x800")
    root.state("zoomed")

    # Attempt to load the logo
    try:
        img = PhotoImage(file=r"icons/logo.png")
        # lang_icon = Image.open("icons/language.png")
        # lang_icon = lang_icon.resize((10, 10))
        # tk_lang_icon = ImageTk.PhotoImage(lang_icon)
        root.iconphoto(False, img)
        # Creating Menubar
        menubar = tk.Menu(root, type="menubar")
        menubar.add_command(label="English", command=lambda: change_language("English"))
        menubar.add_command(label="عربي", command=lambda: change_language("Arabic"))
        root.config(menu=menubar, padx=10, pady=10)

    except Exception as e:
        print(f"Error loading logo: {e}")

    root.title(text_dict[current_language]["title"])
    style = ttk.Style()
    style.theme_use("vista")
    # style.configure("TButton", font= label_font , background= bg_color)
    # style.configure("TLabel", foreground="black", background="red", font= label_font)

    root.configure(background=bg_color)

    cut_var.set("")

    # Create an "Import File" button
    import_button = tk.Button(
        root,
        text=text_dict[current_language]["import_file"],
        width=btn_width,
        font=label_font,
        background=btn_bg_color,
        command=import_file,
    )
    import_button.grid(column=2, row=1, padx=10, pady=10)

    select_font_label, fontChoosen = comboBoxCreation()

    # Create Clear button
    clear_button = tk.Button(
        root,
        text=text_dict[current_language]["clear"],
        width=btn_width,
        font=label_font,
        background=btn_bg_color,
        command=clear_all,
    )
    clear_button.grid(column=3, row=1, padx=10, pady=10)

    # Create Submit button
    submit_button = tk.Button(
        root,
        text=text_dict[current_language]["submit"],
        width=btn_width,
        font=label_font,
        background=btn_bg_color,
        command=start_ocr,
    )
    submit_button.grid(column=4, row=1, padx=10, pady=10)

    # Create a label and entry for Word Space
    cut_label = tk.Label(
        root,
        text=text_dict[current_language]["word_space"],
        font=label_font,
        background=bg_color,
        width=20,
    )
    cut_label.grid(row=3, column=0, pady=10)
    cut_entry = tk.Entry(root, textvariable=cut_var, font=label_font, width=3)

    cut_entry.grid(row=3, column=1, padx=10, pady=10)

    # create save images
    save_img_label = tk.Label(
        root,
        text=text_dict[current_language]["save_img"],
        font=label_font,
        background=bg_color,
        width=20,
        justify="left",
    )
    save_img_label.grid(row=3, column=2, pady=10)
    save_img_check = tk.Checkbutton(
        root,
        variable=save_img_var,
        font=label_font,
        background=bg_color,
        width=2,
        justify="left",
    )
    save_img_check.grid(row=3, column=3, pady=10)

    # Create language combobox
    # language_label = ttk.Label(
    #     root, text=text_dict[current_language]["language"], font=("calibre", 10, "bold")
    # )
    # language_label.grid(column=4, row=3, padx=10, pady=10)
    # language_combo = ttk.Combobox(root, values=["English", "Arabic"])
    # language_combo.grid(column=5, row=3, padx=10, pady=10)
    # language_combo.current(0)
    # language_combo.bind("<<ComboboxSelected>>", change_language)

    # Create text widget
    text_widget.grid(column=1, row=6, columnspan=2, padx=10, pady=10)
    text_widget.tag_configure("tag-right", justify="right", font=text_font)
    # Create a scrollbar
    scroll_bar = tk.Scrollbar(root)

    # Pack the scroll bar
    # Place it to the right side, using tk.RIGHT
    scroll_bar.grid(column=1, row=6, columnspan=2, sticky=N + S + W)
    text_widget.config(yscrollcommand=scroll_bar.set)
    scroll_bar.config(command=text_widget.yview)

    # Run the Tkinter event loop
    root.mainloop()

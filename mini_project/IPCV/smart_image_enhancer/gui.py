"""
gui.py
------
A small Tkinter desktop GUI for the Smart Image Enhancement system.

Lets you:
    1. Browse and load an image
    2. Click "Auto Enhance" to run the smart pipeline
    3. Preview the original vs enhanced image side by side
    4. Save the result

Run with:
    python gui.py
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

import utils
from enhancer import ImageEnhancer

PREVIEW_MAX_SIZE = (380, 380)


class EnhancerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Smart Image Enhancement and Quality Improvement System")
        self.geometry("860x560")
        self.resizable(False, False)

        self.enhancer: ImageEnhancer | None = None
        self.input_path: str | None = None

        self._build_layout()

    # ------------------------------------------------------------------
    def _build_layout(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")

        ttk.Button(top, text="Open Image", command=self.load_image).pack(side="left", padx=4)
        ttk.Button(top, text="Auto Enhance", command=self.run_auto_enhance).pack(side="left", padx=4)
        ttk.Button(top, text="Reset", command=self.reset_image).pack(side="left", padx=4)
        ttk.Button(top, text="Save Result", command=self.save_result).pack(side="left", padx=4)

        images = ttk.Frame(self, padding=10)
        images.pack(fill="both", expand=True)

        left_col = ttk.Frame(images)
        left_col.pack(side="left", expand=True, fill="both", padx=8)
        ttk.Label(left_col, text="Original", font=("Segoe UI", 11, "bold")).pack()
        self.original_label = ttk.Label(left_col, text="No image loaded", anchor="center")
        self.original_label.pack(expand=True, fill="both", pady=6)

        right_col = ttk.Frame(images)
        right_col.pack(side="left", expand=True, fill="both", padx=8)
        ttk.Label(right_col, text="Enhanced", font=("Segoe UI", 11, "bold")).pack()
        self.enhanced_label = ttk.Label(right_col, text="Run Auto Enhance", anchor="center")
        self.enhanced_label.pack(expand=True, fill="both", pady=6)

        self.metrics_text = tk.Text(self, height=8, padx=10, pady=6, state="disabled")
        self.metrics_text.pack(fill="x", padx=10, pady=(0, 10))

    # ------------------------------------------------------------------
    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if not path:
            return
        image = cv2.imread(path)
        if image is None:
            messagebox.showerror("Error", "Could not read that image file.")
            return

        self.input_path = path
        self.enhancer = ImageEnhancer(image)
        self._show_image(self.original_label, image)
        self.enhanced_label.configure(image="", text="Run Auto Enhance")
        self._update_metrics(before=utils.analyze(image))

    def run_auto_enhance(self):
        if self.enhancer is None:
            messagebox.showwarning("No image", "Load an image first.")
            return
        self.enhancer.reset()
        self.enhancer.auto_enhance()
        self._show_image(self.enhanced_label, self.enhancer.result())
        self._update_metrics(
            before=utils.analyze(self.enhancer.original),
            after=utils.analyze(self.enhancer.result()),
            steps=self.enhancer.log,
        )

    def reset_image(self):
        if self.enhancer is None:
            return
        self.enhancer.reset()
        self.enhanced_label.configure(image="", text="Run Auto Enhance")
        self._update_metrics(before=utils.analyze(self.enhancer.original))

    def save_result(self):
        if self.enhancer is None or not self.enhancer.log:
            messagebox.showwarning("Nothing to save", "Run Auto Enhance first.")
            return
        default_name = "enhanced.jpg"
        if self.input_path:
            base, ext = os.path.splitext(os.path.basename(self.input_path))
            default_name = f"{base}_enhanced{ext or '.jpg'}"
        path = filedialog.asksaveasfilename(initialfile=default_name, defaultextension=".jpg")
        if not path:
            return
        self.enhancer.save(path)
        messagebox.showinfo("Saved", f"Enhanced image saved to:\n{path}")

    # ------------------------------------------------------------------
    def _show_image(self, label: ttk.Label, image_bgr):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        pil_img.thumbnail(PREVIEW_MAX_SIZE)
        tk_img = ImageTk.PhotoImage(pil_img)
        label.configure(image=tk_img, text="")
        label.image = tk_img  # keep a reference so it isn't garbage collected

    def _update_metrics(self, before=None, after=None, steps=None):
        self.metrics_text.configure(state="normal")
        self.metrics_text.delete("1.0", tk.END)
        if before:
            self.metrics_text.insert(tk.END, f"BEFORE: {before}\n")
        if after:
            self.metrics_text.insert(tk.END, f"AFTER:  {after}\n")
        if steps:
            self.metrics_text.insert(tk.END, f"Steps applied: {', '.join(steps)}\n")
        self.metrics_text.configure(state="disabled")


if __name__ == "__main__":
    EnhancerApp().mainloop()

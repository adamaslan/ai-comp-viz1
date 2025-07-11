#!/usr/bin/env python3
"""
AI Document Scanner - Optimized for M2 Mac with 8GB RAM
A cross-platform document scanner with AI-powered classification
"""

import sys
import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import torch
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import cv2
import pytesseract
from datetime import datetime
import json
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentScanner:
    """Main application class for the AI Document Scanner"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AI Document Scanner")
        self.root.geometry("1200x800")
        
        # Initialize AI models (lazy loading)
        self.classifier = None
        self.summarizer = None
        
        # Application state
        self.current_image = None
        self.original_image = None  # Keep original for processing
        self.processed_text = ""
        self.document_type = ""
        self.confidence_score = 0.0
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the main user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights for responsive design
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=2)  # Give more space to image
        main_frame.columnconfigure(2, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="AI Document Scanner", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Left panel - Controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        control_frame.columnconfigure(0, weight=1)
        
        # Buttons
        ttk.Button(control_frame, text="Select Image", 
                  command=self.select_image).grid(row=0, column=0, pady=5, sticky=tk.W+tk.E)
        ttk.Button(control_frame, text="Scan Document", 
                  command=self.scan_document).grid(row=1, column=0, pady=5, sticky=tk.W+tk.E)
        ttk.Button(control_frame, text="Export Results", 
                  command=self.export_results).grid(row=2, column=0, pady=5, sticky=tk.W+tk.E)
        
        # Document type display
        ttk.Label(control_frame, text="Document Type:").grid(row=3, column=0, sticky=tk.W, pady=(20, 5))
        self.type_var = tk.StringVar(value="Not analyzed")
        type_label = ttk.Label(control_frame, textvariable=self.type_var, 
                              font=('Arial', 10, 'bold'))
        type_label.grid(row=4, column=0, sticky=tk.W)
        
        # Confidence score
        ttk.Label(control_frame, text="Confidence:").grid(row=5, column=0, sticky=tk.W, pady=(10, 5))
        self.confidence_var = tk.StringVar(value="0%")
        confidence_label = ttk.Label(control_frame, textvariable=self.confidence_var)
        confidence_label.grid(row=6, column=0, sticky=tk.W)
        
        # Image info
        ttk.Label(control_frame, text="Image Info:").grid(row=7, column=0, sticky=tk.W, pady=(10, 5))
        self.image_info_var = tk.StringVar(value="No image loaded")
        info_label = ttk.Label(control_frame, textvariable=self.image_info_var, 
                              font=('Arial', 8))
        info_label.grid(row=8, column=0, sticky=tk.W)
        
        # Progress bar
        self.progress = ttk.Progressbar(control_frame, mode='indeterminate')
        self.progress.grid(row=9, column=0, sticky=tk.W+tk.E, pady=(20, 0))
        
        # Middle panel - Image display with scroll
        image_frame = ttk.LabelFrame(main_frame, text="Document Image", padding="10")
        image_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)
        
        # Create scrollable image canvas
        self.setup_image_canvas(image_frame)
        
        # Right panel - Text output
        text_frame = ttk.LabelFrame(main_frame, text="Extracted Text", padding="10")
        text_frame.grid(row=1, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        
        # Text display with scrollbar
        text_scroll_frame = ttk.Frame(text_frame)
        text_scroll_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        text_scroll_frame.columnconfigure(0, weight=1)
        text_scroll_frame.rowconfigure(0, weight=1)
        
        self.text_display = tk.Text(text_scroll_frame, wrap=tk.WORD, width=40, height=20)
        text_scrollbar = ttk.Scrollbar(text_scroll_frame, orient=tk.VERTICAL, command=self.text_display.yview)
        self.text_display.configure(yscrollcommand=text_scrollbar.set)
        
        self.text_display.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        text_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))

    def setup_image_canvas(self, parent):
        """Setup scrollable image canvas that handles any image dimensions"""
        # Create canvas with scrollbars
        canvas_frame = ttk.Frame(parent)
        canvas_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        
        self.image_canvas = tk.Canvas(canvas_frame, bg='white')
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.image_canvas.xview)
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.image_canvas.yview)
        
        self.image_canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        # Grid layout
        self.image_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Bind mouse wheel for zooming
        self.image_canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.image_canvas.bind("<Button-4>", self.on_mousewheel)
        self.image_canvas.bind("<Button-5>", self.on_mousewheel)
        
        # Bind canvas resize
        self.image_canvas.bind("<Configure>", self.on_canvas_configure)
        
        # Scale factor for zoom
        self.scale_factor = 1.0
        
    def on_mousewheel(self, event):
        """Handle mouse wheel for zooming"""
        if self.current_image is None:
            return
            
        # Determine zoom direction
        if event.delta > 0 or event.num == 4:
            zoom_factor = 1.1
        else:
            zoom_factor = 0.9
            
        self.scale_factor *= zoom_factor
        self.scale_factor = max(0.1, min(5.0, self.scale_factor))  # Limit zoom range
        
        self.display_image(self.original_image)
        
    def on_canvas_configure(self, event):
        """Handle canvas resize"""
        if self.current_image is None:
            return
        # Update scroll region
        self.image_canvas.configure(scrollregion=self.image_canvas.bbox("all"))
        
    def load_lightweight_models(self):
        """Load lightweight AI models optimized for M2 Mac"""
        try:
            self.status_var.set("Loading AI models...")
            self.progress.start()
            
            # Use lightweight models that work well on M2 Mac
            # Set device to CPU for M2 compatibility
            device = "cpu"  # M2 Macs work better with CPU inference for small models
            
            # Load a smaller, efficient classifier
            logger.info("Loading lightweight document classifier...")
            self.classifier = pipeline(
                "zero-shot-classification",
                model="microsoft/DialoGPT-medium",  # Smaller alternative
                device=device
            )
            
            # Alternative: Use a simple rule-based classifier for even better performance
            # self.use_rule_based_classifier = True
            
            self.progress.stop()
            self.status_var.set("Models loaded successfully")
            logger.info("AI models loaded successfully")
            
        except Exception as e:
            self.progress.stop()
            # Fallback to rule-based classification
            self.classifier = None
            self.status_var.set("Using rule-based classification")
            logger.info("Using rule-based classification as fallback")
    
    def select_image(self):
        """Open file dialog to select an image"""
        file_types = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.gif *.webp"),
            ("All files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select Document Image",
            filetypes=file_types
        )
        
        if file_path:
            try:
                # Load image using PIL first for better format support
                pil_image = Image.open(file_path)
                # Convert to RGB if needed
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                
                # Convert to OpenCV format
                self.original_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                
                if self.original_image is None:
                    raise ValueError("Could not load image")
                
                # Reset scale factor
                self.scale_factor = 1.0
                
                # Display image
                self.display_image(self.original_image)
                
                # Update image info
                height, width = self.original_image.shape[:2]
                file_size = os.path.getsize(file_path) / 1024  # KB
                self.image_info_var.set(f"{width}x{height} px\n{file_size:.1f} KB")
                
                self.status_var.set(f"Image loaded: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Image Loading Error", f"Failed to load image: {str(e)}")
                logger.error(f"Error loading image: {e}")
    
    def display_image(self, cv_image):
        """Display OpenCV image in tkinter canvas with proper scaling"""
        try:
            if cv_image is None:
                return
                
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # Apply scaling
            height, width = rgb_image.shape[:2]
            new_width = int(width * self.scale_factor)
            new_height = int(height * self.scale_factor)
            
            # Resize image
            if self.scale_factor != 1.0:
                resized_image = cv2.resize(rgb_image, (new_width, new_height), 
                                         interpolation=cv2.INTER_AREA if self.scale_factor < 1.0 else cv2.INTER_CUBIC)
            else:
                resized_image = rgb_image
            
            # Convert to PIL and then to PhotoImage
            pil_image = Image.fromarray(resized_image)
            photo = ImageTk.PhotoImage(pil_image)
            
            # Clear canvas and display image
            self.image_canvas.delete("all")
            self.image_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.image_canvas.image = photo  # Keep a reference
            
            # Update scroll region
            self.image_canvas.configure(scrollregion=(0, 0, new_width, new_height))
            
            # Store current image for processing
            self.current_image = cv_image
            
        except Exception as e:
            logger.error(f"Error displaying image: {e}")
    
    def scan_document(self):
        """Process the document image and extract text"""
        if self.current_image is None:
            messagebox.showwarning("No Image", "Please select an image first")
            return
        
        try:
            self.status_var.set("Processing document...")
            self.progress.start()
            
            # Preprocess image for better OCR
            processed_image = self.preprocess_image(self.current_image)
            
            # Extract text using OCR
            self.status_var.set("Extracting text...")
            
            # Use better OCR configuration for different image types
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,!?@#$%^&*()_+-=[]{}|;":/<>?`~'
            text = pytesseract.image_to_string(processed_image, config=custom_config)
            
            if not text.strip():
                messagebox.showwarning("No Text", "No text could be extracted from the image")
                self.progress.stop()
                self.status_var.set("No text found")
                return
            
            self.processed_text = text
            self.text_display.delete(1.0, tk.END)
            self.text_display.insert(1.0, text)
            
            # Classify document type
            self.status_var.set("Classifying document...")
            self.classify_document(text)
            
            self.progress.stop()
            self.status_var.set("Document processed successfully")
            
        except Exception as e:
            self.progress.stop()
            self.status_var.set(f"Error processing document: {str(e)}")
            logger.error(f"Error processing document: {e}")
            messagebox.showerror("Processing Error", f"Failed to process document: {str(e)}")
    
    def preprocess_image(self, image):
        """Preprocess image for better OCR results - optimized for various image dimensions"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize if image is too large (for memory efficiency)
        height, width = gray.shape
        max_dimension = 3000
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def classify_document(self, text):
        """Classify the document type using rule-based approach or lightweight AI"""
        try:
            # Rule-based classification (fast and memory efficient)
            self.document_type, self.confidence_score = self.rule_based_classification(text)
            
            # If we have a lightweight classifier loaded, use it for refinement
            if self.classifier:
                try:
                    candidate_labels = [
                        "invoice", "receipt", "contract", "letter", "resume", 
                        "report", "form", "certificate", "license", "manual"
                    ]
                    
                    # Truncate text for efficiency
                    text_sample = text[:500] if len(text) > 500 else text
                    
                    result = self.classifier(text_sample, candidate_labels)
                    
                    # Use AI result if confidence is high
                    if result['scores'][0] > 0.7:
                        self.document_type = result['labels'][0]
                        self.confidence_score = result['scores'][0]
                        
                except Exception as e:
                    logger.warning(f"AI classification failed, using rule-based: {e}")
            
            # Update UI
            self.type_var.set(self.document_type.title())
            self.confidence_var.set(f"{self.confidence_score:.1%}")
            
        except Exception as e:
            logger.error(f"Error classifying document: {e}")
            self.type_var.set("Classification failed")
            self.confidence_var.set("0%")
    
    def rule_based_classification(self, text):
        """Fast rule-based document classification"""
        text_lower = text.lower()
        
        # Define keywords for different document types
        keywords = {
            "invoice": ["invoice", "bill", "amount due", "total", "tax", "subtotal"],
            "receipt": ["receipt", "paid", "change", "cash", "credit card"],
            "contract": ["agreement", "contract", "terms", "conditions", "party"],
            "letter": ["dear", "sincerely", "regards", "yours truly"],
            "resume": ["experience", "education", "skills", "objective", "cv"],
            "report": ["summary", "analysis", "findings", "conclusion", "recommendation"],
            "form": ["form", "application", "please fill", "submit", "required"],
            "certificate": ["certificate", "awarded", "completion", "achievement"],
            "license": ["license", "permit", "authorized", "valid until"],
            "manual": ["manual", "instructions", "step", "procedure", "guide"]
        }
        
        # Score each document type
        scores = {}
        for doc_type, words in keywords.items():
            score = sum(1 for word in words if word in text_lower)
            scores[doc_type] = score / len(words)  # Normalize by number of keywords
        
        # Get best match
        best_type = max(scores, key=scores.get)
        confidence = scores[best_type]
        
        # If confidence is too low, classify as "document"
        if confidence < 0.1:
            return "document", 0.5
        
        return best_type, min(confidence * 2, 1.0)  # Scale confidence
    
    def export_results(self):
        """Export the processed results to a file"""
        if not self.processed_text:
            messagebox.showwarning("No Results", "No processed text to export")
            return
        
        try:
            file_path = filedialog.asksaveasfilename(
                title="Export Results",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("Text files", "*.txt"), ("All files", "*.*")]
            )
            
            if file_path:
                if file_path.endswith('.json'):
                    # Export as JSON
                    results = {
                        "timestamp": datetime.now().isoformat(),
                        "document_type": self.document_type,
                        "confidence_score": self.confidence_score,
                        "extracted_text": self.processed_text,
                        "text_length": len(self.processed_text),
                        "word_count": len(self.processed_text.split()),
                        "image_info": self.image_info_var.get().replace('\n', ' ')
                    }
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
                else:
                    # Export as text
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(f"Document Type: {self.document_type}\n")
                        f.write(f"Confidence: {self.confidence_score:.1%}\n")
                        f.write(f"Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Image Info: {self.image_info_var.get().replace(chr(10), ' ')}\n")
                        f.write("-" * 50 + "\n")
                        f.write(self.processed_text)
                
                self.status_var.set(f"Results exported to {os.path.basename(file_path)}")
                messagebox.showinfo("Export Success", f"Results exported to {file_path}")
                
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            messagebox.showerror("Export Error", f"Failed to export results: {str(e)}")
    
    def run(self):
        """Start the application"""
        # Load models after UI is ready (optional)
        self.root.after(1000, self.load_lightweight_models)
        self.root.mainloop()

def main():
    """Main entry point"""
    try:
        app = DocumentScanner()
        app.run()
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"Error starting application: {e}")

if __name__ == "__main__":
    main()
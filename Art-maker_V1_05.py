import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os 

# --- 1. GÖRSEL İŞLEME FONKSİYONLARI (OpenCV) ---

def gorseli_renk_azalt(gorsel_yolu, parca_sayisi):
    try:
        img = cv2.imread(gorsel_yolu)
        if img is None:
            return None, None
        
        MAX_SIZE = 800
        h, w = img.shape[:2]
        if max(h, w) > MAX_SIZE:
            oran = MAX_SIZE / max(h, w)
            img = cv2.resize(img, None, fx=oran, fy=oran, interpolation=cv2.INTER_AREA)

        data = np.float32(img.reshape((-1, 3)))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        
        _, labels, centers = cv2.kmeans(data, parca_sayisi, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        
        quantized_img = centers[labels.flatten()].reshape(img.shape)
        
        return quantized_img, centers

    except Exception as e:
        print(f"Görsel işleme hatası: {e}")
        return None, None

def numarali_referans_olustur(quantized_img, centers, parca_sayisi, min_alan_orani, cizgi_kalinligi, yazi_boyutu, yazi_kalinligi):
    h, w = quantized_img.shape[:2]
    numbered_img = quantized_img.copy()
    edge_reference = np.full((h, w, 3), 255, dtype=np.uint8) 
    renk_numara_eslesmesi = {} 
    
    min_contour_area = h * w * min_alan_orani

    for i in range(parca_sayisi):
        renk_kodu_bgr = tuple(centers[i])
        numara = i + 1
        renk_numara_eslesmesi[numara] = renk_kodu_bgr 
        
        renk_kodu_array = np.array(renk_kodu_bgr, dtype=np.uint8) 
        mask = cv2.inRange(quantized_img, renk_kodu_array, renk_kodu_array)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < min_contour_area:
                continue
            
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                yazi_rengi = (0, 0, 0) if np.mean(renk_kodu_bgr) > 127 else (255, 255, 255)

                # Numaralı Görsele Sayı Ekle 
                cv2.putText(numbered_img, str(numara), (cX - 10, cY + 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, yazi_boyutu, yazi_rengi, yazi_kalinligi, cv2.LINE_AA)
                            
                # Çizgisel Referansa Sayı Ekle 
                cv2.putText(edge_reference, str(numara), (cX - 10, cY + 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, yazi_boyutu, (0, 0, 0), yazi_kalinligi, cv2.LINE_AA)
            
            # Çizgisel Referansa Kontürleri Çiz
            cv2.drawContours(edge_reference, [contour], -1, (0, 0, 0), cizgi_kalinligi) 

    return numbered_img, edge_reference, renk_numara_eslesmesi

def palet_olustur(centers, renk_numara_eslesmesi):
    KARE_BOYUT = 100
    ARA_BOŞLUK = 20
    FONT_SCALE = 1.0
    FONT_KALINLIK = 2
    parca_sayisi = len(centers)
    
    # DİKEY PALET BOYUTLARI
    palet_genislik = KARE_BOYUT + 2 * ARA_BOŞLUK
    palet_yukseklik = parca_sayisi * KARE_BOYUT + (parca_sayisi + 1) * ARA_BOŞLUK
    
    palet_img = np.full((palet_yukseklik, palet_genislik, 3), 255, dtype=np.uint8)

    for i in range(parca_sayisi):
        numara = i + 1
        bgr_renk_np = centers[i]
        bgr_renk = (int(bgr_renk_np[0]), int(bgr_renk_np[1]), int(bgr_renk_np[2]))
        
        # DİKEY KONUMLANDIRMA
        x1 = ARA_BOŞLUK
        y1 = ARA_BOŞLUK + i * (KARE_BOYUT + ARA_BOŞLUK)
        x2 = x1 + KARE_BOYUT
        y2 = y1 + KARE_BOYUT

        cv2.rectangle(palet_img, (x1, y1), (x2, y2), bgr_renk, -1) 
        cv2.rectangle(palet_img, (x1, y1), (x2, y2), (0, 0, 0), 2)

        metin = str(numara)
        (text_width, text_height), _ = cv2.getTextSize(metin, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_KALINLIK)
        
        text_x = x1 + (KARE_BOYUT - text_width) // 2
        text_y = y1 + (KARE_BOYUT + text_height) // 2
        
        yazi_rengi = (0, 0, 0) if np.mean(bgr_renk) > 127 else (255, 255, 255)
             
        cv2.putText(palet_img, metin, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, yazi_rengi, FONT_KALINLIK, cv2.LINE_AA)

    return palet_img

# --- 2. TKINTER SINIFI (Uygulama Mantığı) ---

class PaintByNumberApp:
    def __init__(self, master):
        self.master = master
        master.title("Python Sayılarla Boyama Hazırlayıcı")
        
        self.gorsel_yolu = None
        self.parca_sayisi = tk.IntVar(value=7) 
        self.zoom_factor = 1.0 
        self.pan_start_x = None
        self.pan_start_y = None
        
        # Slider Değerleri
        self.min_alan_orani = tk.DoubleVar(value=0.0005) # Min kontür alan oranı
        # DEĞİŞTİRİLDİ: Min 0 olarak ayarlandı.
        self.cizgi_kalinligi = tk.IntVar(value=1)       # Kontür çizgisi kalınlığı (0-5)
        self.yazi_boyutu = tk.DoubleVar(value=0.7)      # Yazı boyutu (fontScale)
        self.yazi_kalinligi = tk.IntVar(value=2)        # Yazı kalınlığı (thickness)
        
        # Diğer değişkenler
        self.centers = None
        self.islenmis_img_bgr = None
        self.orijinal_pil_img = None
        self.palet_img_bgr = None
        
        self.ana_cerceve = tk.Frame(master, padx=10, pady=10)
        self.ana_cerceve.pack(pady=20)
        
        # Ana GUI elemanları 
        tk.Button(self.ana_cerceve, text="1. Fotoğrafı Seç", command=self.fotografi_sec).pack(pady=5)
        self.lbl_yol = tk.Label(self.ana_cerceve, text="Fotoğraf Seçilmedi.")
        self.lbl_yol.pack(pady=5)
        tk.Label(self.ana_cerceve, text="2. Kaç Renk Olsun (6-12 arası önerilir):").pack(pady=5)
        self.ent_sayi = tk.Entry(self.ana_cerceve, textvariable=self.parca_sayisi, width=10, justify='center')
        self.ent_sayi.pack(pady=5)
        self.btn_onizle = tk.Button(self.ana_cerceve, text="3. Ön İzleme Başlat", command=self.onizleme_goster, state=tk.DISABLED)
        self.btn_onizle.pack(pady=20)

    def fotografi_sec(self):
        yol = filedialog.askopenfilename(
            title="Bir Fotoğraf Dosyası Seçin",
            filetypes=[("Görüntü Dosyaları", "*.jpg *.jpeg *.png *.bmp")]
        )
        if yol:
            self.gorsel_yolu = yol
            self.lbl_yol.config(text=f"Seçilen: ...{os.path.basename(yol)}") 
            self.btn_onizle.config(state=tk.NORMAL)
            self.zoom_factor = 1.0 

    def kapat_onizleme(self):
        """Önizleme penceresini kapatır."""
        if hasattr(self, 'onizleme_penceresi') and self.onizleme_penceresi.winfo_exists():
            self.onizleme_penceresi.destroy()

    def onizleme_goster(self):
        if not self.gorsel_yolu:
            messagebox.showerror("Hata", "Lütfen önce bir fotoğraf seçin.")
            return

        try:
            parca_sayisi = self.parca_sayisi.get()
            if parca_sayisi < 2 or parca_sayisi > 30:
                 messagebox.showwarning("Uyarı", "Renk sayısı 2 ile 30 arasında olmalıdır.")
                 return
        except tk.TclError:
            messagebox.showerror("Hata", "Lütfen geçerli bir sayı girin.")
            return
        
        islenmis_img_bgr, centers = gorseli_renk_azalt(self.gorsel_yolu, parca_sayisi)
        
        if islenmis_img_bgr is None:
            messagebox.showerror("Hata", "Görsel işlenirken bir sorun oluştu.")
            return
            
        self.centers = centers 
        self.islenmis_img_bgr = islenmis_img_bgr 
        self.orijinal_pil_img = Image.fromarray(cv2.cvtColor(islenmis_img_bgr, cv2.COLOR_BGR2RGB))
        self.palet_img_bgr = palet_olustur(self.centers, {}) 

        # Önceki pencereler varsa kapat
        self.kapat_onizleme()
            
        self.onizleme_penceresi = tk.Toplevel(self.master)
        self.onizleme_penceresi.title(f"Interaktif Referanslar ({parca_sayisi} Renk)")
        self.onizleme_penceresi.protocol("WM_DELETE_WINDOW", self.kapat_onizleme)
        
        # --- ÜÇ BÖLMELİ ÇERÇEVE ---
        main_view_frame = tk.Frame(self.onizleme_penceresi)
        main_view_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 1. Sol Taraf: Çizgisel Referans (Trace Map)
        left_frame = tk.Frame(main_view_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        tk.Label(left_frame, text="1. Trace Map").pack(pady=5)
        self.canvas_trace = tk.Canvas(left_frame, bg="gray80")
        self.canvas_trace.pack(fill=tk.BOTH, expand=True)

        # 2. Orta Taraf: Renkli Referans
        middle_frame = tk.Frame(main_view_frame)
        middle_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        tk.Label(middle_frame, text="2. Renkli Görsel").pack(pady=5)
        self.canvas_color = tk.Canvas(middle_frame, bg="gray80")
        self.canvas_color.pack(fill=tk.BOTH, expand=True)
        
        # 3. Sağ Taraf: Dikey Palet
        right_frame = tk.Frame(main_view_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5) 
        tk.Label(right_frame, text="3. Palet").pack(pady=5)
        self.canvas_palet = tk.Canvas(right_frame, bg="white")
        self.canvas_palet.pack(fill=tk.Y, expand=True)

        # --- SCROLLBAR'LAR VE BİNDİNG'LER ---
        
        # Scrollbar'lar (Hepsine etki eder)
        self.vsb = tk.Scrollbar(self.onizleme_penceresi, orient="vertical", command=self.scroll_y_command)
        self.hsb = tk.Scrollbar(self.onizleme_penceresi, orient="horizontal", command=self.scroll_x_command)
        self.vsb.pack(side="right", fill="y")
        self.hsb.pack(side="bottom", fill="x")
        
        self.canvas_trace.config(yscrollcommand=self.vsb.set, xscrollcommand=self.hsb.set)
        self.canvas_color.config(yscrollcommand=self.vsb.set, xscrollcommand=self.hsb.set)
        self.canvas_palet.config(yscrollcommand=self.vsb.set, xscrollcommand=self.hsb.set) 
        
        # Tekerlek Tuşu (Button-2, pan modu) için Binding (Sadece sol/orta canvas'a bağlıyoruz)
        self.canvas_trace.bind("<ButtonPress-2>", self.start_pan)
        self.canvas_trace.bind("<B2-Motion>", self.pan_move)
        self.canvas_trace.bind("<ButtonRelease-2>", self.stop_pan)
        
        self.canvas_color.bind("<ButtonPress-2>", self.start_pan)
        self.canvas_color.bind("<B2-Motion>", self.pan_move)
        self.canvas_color.bind("<ButtonRelease-2>", self.stop_pan)
        
        # Zoom için Binding (Tekerlek yukarı/aşağı)
        self.canvas_trace.bind("<MouseWheel>", self.on_mousewheel) 
        self.canvas_trace.bind("<Button-4>", self.on_mousewheel) 
        self.canvas_trace.bind("<Button-5>", self.on_mousewheel) 
        
        self.guncel_gorselleri_isle()
        self.update_image_on_canvas() 
        
        # --- SLIDER KONTROL PANELİ (Alt Kısım) ---
        slider_frame = tk.Frame(self.onizleme_penceresi, pady=5)
        slider_frame.pack(fill=tk.X)
        
        # SLIDER SATIRI 1: Kontür Ayarları
        slider_row1 = tk.Frame(slider_frame)
        slider_row1.pack(fill=tk.X)

        # 1. Kontür Kalınlığı Slider'ı
        # GÜNCELLENDİ: from_=0 olarak ayarlandı
        tk.Label(slider_row1, text="Çizgi Kalınlığı (0-5):").pack(side=tk.LEFT, padx=10)
        tk.Scale(slider_row1, from_=0, to=5, orient=tk.HORIZONTAL, variable=self.cizgi_kalinligi, 
                 command=self.on_slider_change, length=150).pack(side=tk.LEFT, padx=10)
        
        # 2. Minimum Kontür Alanı Slider'ı (Gürültü Atma)
        tk.Label(slider_row1, text="Min Alan Oranı:").pack(side=tk.LEFT, padx=10)
        tk.Scale(slider_row1, from_=0.0001, to=0.005, resolution=0.0001, orient=tk.HORIZONTAL, 
                 variable=self.min_alan_orani, command=self.on_slider_change, length=150).pack(side=tk.LEFT, padx=10)
                 
        # SLIDER SATIRI 2: Yazı Ayarları
        slider_row2 = tk.Frame(slider_frame)
        slider_row2.pack(fill=tk.X)
        
        # 3. Yazı Boyutu Slider'ı
        tk.Label(slider_row2, text="Yazı Boyutu (0.1-2.0):").pack(side=tk.LEFT, padx=10)
        tk.Scale(slider_row2, from_=0.1, to=2.0, resolution=0.1, orient=tk.HORIZONTAL, 
                 variable=self.yazi_boyutu, command=self.on_slider_change, length=150).pack(side=tk.LEFT, padx=10)

        # 4. Yazı Kalınlığı Slider'ı
        tk.Label(slider_row2, text="Yazı Kalınlığı (1-5):").pack(side=tk.LEFT, padx=10)
        tk.Scale(slider_row2, from_=1, to=5, resolution=1, orient=tk.HORIZONTAL, 
                 variable=self.yazi_kalinligi, command=self.on_slider_change, length=150).pack(side=tk.LEFT, padx=10)

        
        # Onay Butonları
        btn_frame = tk.Frame(self.onizleme_penceresi)
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="4. Onayla ve Referansları Kaydet", 
                  command=lambda: self.kaydetme_islemini_baslat(self.islenmis_img_bgr, self.centers, parca_sayisi),
                  bg='green', fg='white').pack(side=tk.LEFT, padx=20)

        tk.Button(btn_frame, text="Yeni Renk Sayısı Dene", command=self.yeni_renk_sayisi_gir).pack(side=tk.LEFT, padx=20)
        
        self.onizleme_penceresi.update_idletasks()
        
    def scroll_y_command(self, *args):
        """Dikey kaydırma komutunu tüm canvas'lara uygular."""
        self.canvas_trace.yview(*args)
        self.canvas_color.yview(*args)
        self.canvas_palet.yview(*args)

    def scroll_x_command(self, *args):
        """Yatay kaydırma komutunu tüm canvas'lara uygular."""
        self.canvas_trace.xview(*args)
        self.canvas_color.xview(*args)
        self.canvas_palet.xview(*args)
        
    # --- PAN (KAYDIRMA) Fonksiyonları ---
    def start_pan(self, event):
        """Tekerlek tuşuna basıldığında kaydırma modunu başlatır."""
        self.pan_start_x = event.x
        self.pan_start_y = event.y

    def pan_move(self, event):
        """Kaydırma modunda farenin hareketine göre canvas'ları kaydırır."""
        if self.pan_start_x is not None and self.pan_start_y is not None:
            # Kaydırma miktarı (piksel)
            dx = event.x - self.pan_start_x
            dy = event.y - self.pan_start_y
            
            # Kaydırma miktarını Canvas'ın boyutuna göre oransal olarak ayarla
            
            # YATAY KAYDIRMA
            if dx != 0:
                self.canvas_trace.xview("scroll", -dx, "units")
                self.canvas_color.xview("scroll", -dx, "units")
                self.canvas_palet.xview("scroll", -dx, "units")

            # DİKEY KAYDIRMA
            if dy != 0:
                self.canvas_trace.yview("scroll", -dy, "units")
                self.canvas_color.yview("scroll", -dy, "units")
                self.canvas_palet.yview("scroll", -dy, "units")

            # Yeni başlangıç pozisyonunu güncelle
            self.pan_start_x = event.x
            self.pan_start_y = event.y

    def stop_pan(self, event):
        """Tekerlek tuşu bırakıldığında kaydırma modunu sonlandırır."""
        self.pan_start_x = None
        self.pan_start_y = None
    # --- PAN (KAYDIRMA) Fonksiyonları Sonu ---

    def on_slider_change(self, value):
        self.guncel_gorselleri_isle()
        self.update_image_on_canvas()

    def guncel_gorselleri_isle(self):
        """Mevcut slider ayarlarına göre çizgisel görseli yeniden işler."""
        if self.islenmis_img_bgr is None or self.centers is None:
            return

        min_alan = self.min_alan_orani.get()
        cizgi_kalinligi = self.cizgi_kalinligi.get()
        yazi_boyutu = self.yazi_boyutu.get()
        yazi_kalinligi = self.yazi_kalinligi.get() 
        
        # Güncel ayarlar ile çizgisel referansı yeniden oluştur
        self.numbered_img_bgr, self.edge_pil_img_bgr, _ = numarali_referans_olustur(
            self.islenmis_img_bgr, self.centers, len(self.centers), min_alan, cizgi_kalinligi, yazi_boyutu, yazi_kalinligi
        )
        
        self.trace_pil_img = Image.fromarray(cv2.cvtColor(self.edge_pil_img_bgr, cv2.COLOR_BGR2RGB))
        self.color_pil_img = Image.fromarray(cv2.cvtColor(self.numbered_img_bgr, cv2.COLOR_BGR2RGB)) 

    def on_mousewheel(self, event):
        """Fare tekerleği hareketi ile zoom yapar."""
        if event.num == 4 or event.delta > 0: 
            self.zoom_factor *= 1.1
        elif event.num == 5 or event.delta < 0: 
            self.zoom_factor /= 1.1
        
        if self.zoom_factor < 0.1:
            self.zoom_factor = 0.1
            
        self.update_image_on_canvas()

    def update_image_on_canvas(self):
        """Güncel zoom faktörüne göre tüm görselleri yeniden boyutlandırır ve canvas'ları günceller."""
        if self.orijinal_pil_img is None or self.trace_pil_img is None or self.palet_img_bgr is None:
            return

        # Görsel boyutları
        orijinal_w = self.orijinal_pil_img.width
        orijinal_h = self.orijinal_pil_img.height
        
        yeni_w = int(orijinal_w * self.zoom_factor)
        yeni_h = int(orijinal_h * self.zoom_factor)
        
        # Palet boyutları
        palet_pil_img = Image.fromarray(cv2.cvtColor(self.palet_img_bgr, cv2.COLOR_BGR2RGB))
        palet_w = palet_pil_img.width
        palet_h = palet_pil_img.height
        
        # Yalnızca paletin YÜKSEKLİĞİNİ ana görselin yüksekliği ile orantılı zoom yapıyoruz.
        palet_yeni_h = int(palet_h * self.zoom_factor)
        palet_yeni_w = palet_w 

        
        # 1. Çizgisel Görseli Güncelle (Trace Map)
        resized_trace_img = self.trace_pil_img.resize((yeni_w, yeni_h), Image.LANCZOS)
        self.img_tk_trace = ImageTk.PhotoImage(resized_trace_img)
        self.canvas_trace.delete("all")
        self.canvas_trace.create_image(0, 0, anchor="nw", image=self.img_tk_trace)
        
        # 2. Renkli Görseli Güncelle
        resized_color_img = self.color_pil_img.resize((yeni_w, yeni_h), Image.LANCZOS)
        self.img_tk_color = ImageTk.PhotoImage(resized_color_img)
        self.canvas_color.delete("all")
        self.canvas_color.create_image(0, 0, anchor="nw", image=self.img_tk_color)
        
        # 3. Palet Görselini Güncelle
        resized_palet_img = palet_pil_img.resize((palet_yeni_w, palet_yeni_h), Image.LANCZOS)
        self.palet_img_tk = ImageTk.PhotoImage(resized_palet_img)
        self.canvas_palet.delete("all")
        self.canvas_palet.create_image(0, 0, anchor="nw", image=self.palet_img_tk)
        
        # Tüm canvas'ların kaydırma alanını görsellerin boyutuna ayarla
        self.canvas_trace.config(scrollregion=(0, 0, yeni_w, yeni_h))
        self.canvas_color.config(scrollregion=(0, 0, yeni_w, yeni_h))
        self.canvas_palet.config(scrollregion=(0, 0, palet_yeni_w, palet_yeni_h))
        
        # Palet canvas'ının genişliğini her zaman güncel tutmak için
        self.canvas_palet.config(width=palet_yeni_w)

    def yeni_renk_sayisi_gir(self):
        self.kapat_onizleme() 
            
        yeni_sayi = simpledialog.askinteger(
            "Renk Sayısı Girişi", 
            "Yeni renk sayısını girin (Örn: 6, 8, 10):", 
            parent=self.master, 
            minvalue=2, 
            maxvalue=30,
            initialvalue=self.parca_sayisi.get()
        )
        
        if yeni_sayi is not None:
            self.parca_sayisi.set(yeni_sayi)
            self.ent_sayi.delete(0, tk.END)
            self.ent_sayi.insert(0, str(yeni_sayi))
            self.zoom_factor = 1.0 
            self.onizleme_goster() 

    def kaydetme_islemini_baslat(self, quantized_img, centers, parca_sayisi):
        # Kayıt için son değerleri al
        min_alan = self.min_alan_orani.get()
        cizgi_kalinligi = self.cizgi_kalinligi.get()
        yazi_boyutu = self.yazi_boyutu.get()
        yazi_kalinligi = self.yazi_kalinligi.get() 
        
        try:
            # Kayıt için son işleme (tüm slider değerleri ile)
            numbered_img, edge_reference, renk_numara_eslesmesi = numarali_referans_olustur(
                quantized_img, centers, parca_sayisi, min_alan, cizgi_kalinligi, yazi_boyutu, yazi_kalinligi
            )
        except Exception as e:
            messagebox.showerror("Hata", f"Numaralandırma sırasında bir hata oluştu: {e}")
            return
            
        try:
            palet_img = palet_olustur(centers, renk_numara_eslesmesi)
        except Exception as e:
            messagebox.showerror("Hata", f"Palet oluşturma sırasında bir hata oluştu: {e}")
            return
        
        base_save_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Dosyaları", "*.png")],
            initialfile=f"boyama_referansi_{parca_sayisi}_renk",
            title="Referans Dosyalarını Kaydetmek İçin Konum Seçin"
        )
        
        if base_save_path:
            base_name = base_save_path.replace('.png', '') 
            
            cv2.imwrite(f"{base_name}_1_numarali.png", numbered_img)
            cv2.imwrite(f"{base_name}_2_cizgisel_ve_numarali.png", edge_reference)
            cv2.imwrite(f"{base_name}_3_palet.png", palet_img)

            print("\n--- KAYDEDİLEN RENK PALETİ VE NUMARALARI ---")
            for numara, bgr in renk_numara_eslesmesi.items():
                r, g, b = bgr[2], bgr[1], bgr[0]
                print(f"Numara {numara}: RGB({r}, {g}, {b})")
            print("-------------------------------------------\n")

            messagebox.showinfo("Başarılı", f"Tüm referans dosyaları ({parca_sayisi} renk) başarıyla kaydedildi!")
            self.kapat_onizleme() 

# --- 3. UYGULAMAYI BAŞLATMA ---

if __name__ == "__main__":
    root = tk.Tk()
    app = PaintByNumberApp(root)
    root.mainloop()
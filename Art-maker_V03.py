import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os 

# --- 1. GÖRSEL İŞLEME FONKSİYONLARI (OpenCV) ---

# gorseli_renk_azalt fonksiyonu burada aynı kalır...
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

def numarali_referans_olustur(quantized_img, centers, parca_sayisi):
    """
    Renkli alanlara numaralar ekler, kontürleri belirginleştirir ve renk eşleşmesini döndürür.
    Çizgisel referansa da numaralar eklenir.
    """
    h, w = quantized_img.shape[:2]
    numbered_img = quantized_img.copy()
    
    # Numaraların ekleneceği çizgisel referans için beyaz zemin oluştur
    edge_reference = np.full((h, w, 3), 255, dtype=np.uint8) 

    renk_numara_eslesmesi = {} 

    for i in range(parca_sayisi):
        renk_kodu_bgr = tuple(centers[i])
        numara = i + 1
        renk_numara_eslesmesi[numara] = renk_kodu_bgr 
        
        renk_kodu_array = np.array(renk_kodu_bgr, dtype=np.uint8) 
        
        # Sadece bu renge ait pikselleri maskele (Doğrudan K-Means çıktısından)
        mask = cv2.inRange(quantized_img, renk_kodu_array, renk_kodu_array)
        
        # Kontür bulma modunu değiştirmiyoruz (RETR_EXTERNAL ve CHAIN_APPROX_SIMPLE en iyisidir)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Çok küçük gürültü alanlarını atla
            if cv2.contourArea(contour) < (h * w / 500):
                continue
            
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                yazi_rengi = (0, 0, 0) if np.mean(renk_kodu_bgr) > 127 else (255, 255, 255)

                # Numaralı Görsele Sayı Ekle
                cv2.putText(numbered_img, str(numara), (cX - 10, cY + 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, yazi_rengi, 2, cv2.LINE_AA)
                            
                # Çizgisel Referansa Sayı Ekle (Siyah renkte)
                cv2.putText(edge_reference, str(numara), (cX - 10, cY + 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            
            # --- ÇİZGİ KALINLIĞI VE TUTARLILIK DÜZENLEMESİ ---
            # Kontürleri doğrudan çizgisel referansa çiz. Kalınlık 1 tutarlılık sağlar.
            cv2.drawContours(edge_reference, [contour], -1, (0, 0, 0), 1) 

    return numbered_img, edge_reference, renk_numara_eslesmesi

# palet_olustur fonksiyonu burada aynı kalır...
def palet_olustur(centers, renk_numara_eslesmesi):
    KARE_BOYUT = 100
    ARA_BOŞLUK = 20
    FONT_SCALE = 1.0
    FONT_KALINLIK = 2
    parca_sayisi = len(centers)
    palet_genislik = parca_sayisi * KARE_BOYUT + (parca_sayisi + 1) * ARA_BOŞLUK
    palet_yukseklik = KARE_BOYUT + 2 * ARA_BOŞLUK
    palet_img = np.full((palet_yukseklik, palet_genislik, 3), 255, dtype=np.uint8)

    for i in range(parca_sayisi):
        numara = i + 1
        bgr_renk_np = centers[i]
        bgr_renk = (int(bgr_renk_np[0]), int(bgr_renk_np[1]), int(bgr_renk_np[2]))
        
        x1 = ARA_BOŞLUK + i * (KARE_BOYUT + ARA_BOŞLUK)
        y1 = ARA_BOŞLUK
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

# PaintByNumberApp sınıfının tüm metotları (fotografi_sec, onizleme_goster, on_mousewheel,
# update_image_on_canvas, yeni_renk_sayisi_gir, kaydetme_islemini_baslat) 
# önceki haliyle (kaydırma ve zoom özellikli) tam olarak aynı kalır.

class PaintByNumberApp:
    def __init__(self, master):
        self.master = master
        master.title("Python Sayılarla Boyama Hazırlayıcı")
        
        self.gorsel_yolu = None
        self.parca_sayisi = tk.IntVar(value=7) 
        self.zoom_factor = 1.0 
        
        self.ana_cerceve = tk.Frame(master, padx=10, pady=10)
        self.ana_cerceve.pack(pady=20)
        
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
            messagebox.showerror("Hata", "Görsel işlenirken bir sorun oluştu. Dosya bozuk olabilir.")
            return
            
        self.orijinal_pil_img = Image.fromarray(cv2.cvtColor(islenmis_img_bgr, cv2.COLOR_BGR2RGB))
        self.centers = centers 
        self.islenmis_img_bgr = islenmis_img_bgr 

        if hasattr(self, 'onizleme_penceresi') and self.onizleme_penceresi.winfo_exists():
            self.onizleme_penceresi.destroy() 
            
        self.onizleme_penceresi = tk.Toplevel(self.master)
        self.onizleme_penceresi.title(f"Ön İzleme ({parca_sayisi} Renk)")
        
        self.image_frame = tk.Frame(self.onizleme_penceresi)
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.canvas = tk.Canvas(self.image_frame)
        self.vsb = tk.Scrollbar(self.image_frame, orient="vertical", command=self.canvas.yview)
        self.hsb = tk.Scrollbar(self.image_frame, orient="horizontal", command=self.canvas.xview)
        
        self.canvas.configure(yscrollcommand=self.vsb.set, xscrollcommand=self.hsb.set)
        
        self.vsb.pack(side="right", fill="y")
        self.hsb.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)
        
        self.canvas.bind("<MouseWheel>", self.on_mousewheel) 
        self.canvas.bind("<Button-4>", self.on_mousewheel) 
        self.canvas.bind("<Button-5>", self.on_mousewheel) 

        self.update_image_on_canvas() 
        
        tk.Label(self.onizleme_penceresi, text=f"Mevcut renk sayısı: {parca_sayisi}. Onaylıyor musunuz? (Zoom için fare tekerleğini kullanın)").pack(pady=5)
        
        btn_frame = tk.Frame(self.onizleme_penceresi)
        btn_frame.pack(pady=15)

        tk.Button(btn_frame, text="4. Onayla ve Referansları Kaydet", 
                  command=lambda: self.kaydetme_islemini_baslat(self.islenmis_img_bgr, self.centers, parca_sayisi),
                  bg='green', fg='white').pack(side=tk.LEFT, padx=10)

        tk.Button(btn_frame, text="Yeni Renk Sayısı Dene", command=self.yeni_renk_sayisi_gir).pack(side=tk.LEFT, padx=10)
        
        self.onizleme_penceresi.update_idletasks()
    
    def on_mousewheel(self, event):
        if event.num == 4 or event.delta > 0: 
            self.zoom_factor *= 1.1
        elif event.num == 5 or event.delta < 0: 
            self.zoom_factor /= 1.1
        
        if self.zoom_factor < 0.1:
            self.zoom_factor = 0.1
            
        self.update_image_on_canvas()

    def update_image_on_canvas(self):
        if not hasattr(self, 'orijinal_pil_img'):
            return

        yeni_w = int(self.orijinal_pil_img.width * self.zoom_factor)
        yeni_h = int(self.orijinal_pil_img.height * self.zoom_factor)
        
        resized_img = self.orijinal_pil_img.resize((yeni_w, yeni_h), Image.LANCZOS)
        self.img_tk = ImageTk.PhotoImage(resized_img)
        
        self.canvas.delete("all")
        
        self.canvas.create_image(0, 0, anchor="nw", image=self.img_tk)
        
        self.canvas.config(scrollregion=(0, 0, yeni_w, yeni_h))

    def yeni_renk_sayisi_gir(self):
        if hasattr(self, 'onizleme_penceresi'):
            self.onizleme_penceresi.destroy() 
            
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
        try:
            numbered_img, edge_reference, renk_numara_eslesmesi = numarali_referans_olustur(quantized_img, centers, parca_sayisi)
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
            if hasattr(self, 'onizleme_penceresi'):
                self.onizleme_penceresi.destroy() 

# --- 3. UYGULAMAYI BAŞLATMA ---

if __name__ == "__main__":
    root = tk.Tk()
    app = PaintByNumberApp(root)
    root.mainloop()
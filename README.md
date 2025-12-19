# 🇹🇷 Derin Öğrenme ile Türkçe Duygu Analizi (Sentiment Analysis)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Library](https://img.shields.io/badge/Library-HuggingFace%20Transformers-yellow)
![Framework](https://img.shields.io/badge/Framework-PyTorch-red)
![License](https://img.shields.io/badge/License-MIT-green)

Bu proje, Türkçe ürün yorumlarını analiz ederek metinlerin **Olumlu** veya **Olumsuz** duygu durumunu tespit etmek amacıyla geliştirilmiş bir Doğal Dil İşleme (NLP) uygulamasıdır. Proje kapsamında Transformer tabanlı modern mimariler kullanılmış ve modellerin performansı karşılaştırılmıştır.

---

## 📋 İçindekiler
1. [Proje Konusu ve Önemi](#1-proje-konusu-ve-önemi)
2. [Veri Seti](#2-veri-seti)
3. [Yöntem ve Algoritma Seçimi](#3-yöntem-ve-algoritma-seçimi)
4. [Model Eğitimi ve Değerlendirme](#4-model-eğitimi-ve-değerlendirme)
5. [Kurulum ve Kullanım](#5-kurulum-ve-kullanım)
6. [Proje Sunumu (Demo)](#6-proje-sunumu-demo)

---

## 1. Proje Konusu ve Önemi

### Seçilme Gerekçesi
E-ticaret ve dijital hizmetlerin artmasıyla birlikte, kullanıcı geri bildirimlerinin (yorumların) manuel olarak analiz edilmesi imkansız hale gelmiştir. İşletmelerin müşteri memnuniyetini ölçebilmesi ve potansiyel alıcıların doğru karar verebilmesi için bu yorumların otomatik olarak sınıflandırılması gerekmektedir.

### İlgili Alanın Önemi
Duygu analizi (Sentiment Analysis); marka itibar yönetimi, pazar araştırması ve müşteri ilişkileri yönetimi (CRM) için kritik bir öneme sahiptir. Türkçe, sondan eklemeli yapısı nedeniyle NLP çalışmalarında İngilizceye göre daha zorlu bir dildir ve bu alanda özelleşmiş modellere duyulan ihtiyaç her geçen gün artmaktadır.

### Daha Önce Yapılan Uygulamalar
Geleneksel olarak bu alanda Naive Bayes, SVM gibi makine öğrenmesi yöntemleri veya LSTM/GRU gibi tekrarlayan sinir ağları (RNN) kullanılmıştır. Ancak 2017 sonrası ortaya çıkan **Transformer** mimarileri (BERT, RoBERTa vb.), bağlamsal ilişkiyi (context) çok daha iyi yakalayarak bu alanda "State-of-the-Art" (SOTA) sonuçlar vermektedir. Bu proje, bu modern yaklaşımı benimsemektedir.

---

## 2. Veri Seti

Projede, halka açık kaynaklardan derlenen Türkçe ürün yorumları kullanılmıştır.

* **Kaynak:** Proje kapsamında kullanılan veri seti, çeşitli e-ticaret sitelerinden toplanan yorumları içermektedir.
* **Veri Büyüklüğü:** Toplam **8491** adet yorum.
* **Etiketler:**
    * `1`: Olumlu (Positive)
    * `0`: Olumsuz (Negative)
* **Ön İşleme Adımları:**
    * Tüm harfler küçük harfe çevrildi.
    * Noktalama işaretleri ve özel karakterler temizlendi.
    * Türkçe etkisiz kelimeler (Stopwords) `nltk` kütüphanesi kullanılarak çıkarıldı.
    * Veri seti **%80 Eğitim (Train)** ve **%20 Test (Test)** olarak ayrıldı.
    * Veri seti adresi: https://www.kaggle.com/datasets/burhanbilenn/turkish-customer-reviews-for-binary-classification/code

---

## 3. Yöntem ve Algoritma Seçimi

Bu projede **Transfer Learning (Transfer Öğrenme)** yöntemi kullanılmıştır. Sıfırdan bir model eğitmek yerine, büyük veri setleriyle önceden eğitilmiş (Pre-trained) Transformer modelleri, bu spesifik görev (Fine-tuning) için uyarlanmıştır.

### Karşılaştırmalı Analiz ve Seçim

| Yöntem | Avantaj | Dezavantaj | Bu Projedeki Durumu |
| :--- | :--- | :--- | :--- |
| **Geleneksel ML (TF-IDF + SVM)** | Hızlı ve basit. | Bağlamı (context) kaçırır, kelime sırasını önemsemez. | Kullanılmadı. |
| **LSTM / RNN** | Kelime sırasını dikkate alır. | Uzun cümlelerde "Vanishing Gradient" sorunu yaşar, yavaştır. | Kullanılmadı. |
| **BERTurk (dbmdz)** | Türkçe için özel eğitilmiştir, morfolojik yapıyı iyi anlar. | Yüksek işlem gücü gerektirir. | **Seçilen Ana Model** ✅ |
| **XLM-RoBERTa** | Çok dillidir (Multilingual), geniş kelime haznesi vardır. | Tek bir dile özelleşmiş modeller kadar hassas olmayabilir. | Karşılaştırma için kullanıldı. |

**Neden Transformerlar?**
"Attention Mechanism" (Dikkat Mekanizması) sayesinde cümlenin tamamındaki kelimelerin birbirleriyle ilişkisini aynı anda işleyerek, Türkçedeki devrik cümleler veya karmaşık ifadelerde bile yüksek başarı sağlar.

---

## 4. Model Eğitimi ve Değerlendirme

İki farklı model eğitilmiş ve performansları test veri seti üzerinde karşılaştırılmıştır.

### Eğitim Parametreleri
* **Kütüphane:** Hugging Face `transformers`, `Trainer` API
* **Batch Size:** 16
* **Learning Rate:** 2e-5
* **Epoch:** 1 (Overfitting'i önlemek ve hızlı sonuç için)
* **Optimizer:** AdamW

### Performans Sonuçları

| Model | Accuracy (Doğruluk) | F1-Score | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: |
| **BERTurk (dbmdz/bert-base-turkish-cased)** | **0.920** | **0.917** | **0.937** | **0.898** |
| XLM-RoBERTa (xlm-roberta-base) | 0.903 | 0.900 | 0.915 | 0.885 |

**Değerlendirme:**
* **BERTurk**, Türkçe için özel olarak eğitildiği için çok dilli model olan XLM-RoBERTa'ya göre daha yüksek performans göstermiştir.
* **%92** doğruluk oranı, modelin gerçek dünya senaryolarında güvenle kullanılabileceğini göstermektedir.
* Modelin karmaşıklık matrisi (confusion matrix) incelendiğinde, hem olumlu hem de olumsuz sınıfları ayırt etmede dengeli bir başarı sergilediği görülmüştür.

---

## 5. Kurulum ve Kullanım

Projeyi yerel bilgisayarınızda veya Google Colab'de çalıştırmak için aşağıdaki adımları izleyebilirsiniz.

### Gereksinimler
Proje Python 3.x gerektirir. Gerekli kütüphaneleri yüklemek için:


pip install transformers datasets evaluate accelerate scikit-learn seaborn matplotlib nltk torch gradio

### Çalıştırma
1.  Repo'yu klonlayın, zip olarak indirin veya colab linkinden çalıştırın. https://colab.research.google.com/drive/1hmlVd6mE39-21yCN7Yhkks-xfnM2rGCG?usp=sharing
2.  `derin_ogrenme_proje.ipynb` dosyasını Jupyter Notebook veya Google Colab ile açın.
3.  Hücreleri sırasıyla çalıştırın.

---

## 6. Proje Sunumu (Demo)

Projenin son kullanıcı tarafından test edilebilmesi için **Gradio** kütüphanesi kullanılarak interaktif bir web arayüzü geliştirilmiştir.

### Nasıl Kullanılır?
Notebook dosyasının son bölümündeki kod çalıştırıldığında, kullanıcıya bir metin kutusu sunulur. Kullanıcı buraya bir ürün yorumu girer ve model anlık olarak şu çıktıları üretir:
* **Tahmin:** Olumlu 😊 veya Olumsuz 😞
* **Güven Skoru:** Modelin tahmininden ne kadar emin olduğu (Örn: %98.5).

### Örnek Senaryo
> **Girdi:** "Ürün beklediğimden çok daha kaliteli geldi, kargo da hızlıydı."
>
> **Çıktı:** Olumlu (Positive) 😊 - Güven Skoru: %99.42

Bu arayüz, teknik bilgisi olmayan kullanıcıların dahi modeli kolayca deneyimlemesine olanak tanır.

---

### Lisans
Bu proje MIT lisansı ile lisanslanmıştır.

### İletişim
Furkan Topçu

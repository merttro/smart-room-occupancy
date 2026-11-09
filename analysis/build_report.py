"""Build the five-page Turkish internship report from computed results.

Run after analyze.py: python analysis/build_report.py
Requires reportlab. Does not evaluate, train, or modify any model.
"""
import argparse
import html
import json
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "analysis" / "results"
REPO_URL = "https://github.com/merttro/smart-room-occupancy"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "report")
    parser.add_argument("--font", type=Path)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    candidates = [args.font, Path("C:/Windows/Fonts/arial.ttf"),
                  Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]
    font = next((p for p in candidates if p and p.exists()), None)
    if font is None:
        raise SystemExit("Pass --font with a Unicode TrueType font path.")
    bold_candidates = [font.with_name("arialbd.ttf"), font.with_name("DejaVuSans-Bold.ttf")]
    bold = next((p for p in bold_candidates if p.exists()), font)
    pdfmetrics.registerFont(TTFont("Body", str(font)))
    pdfmetrics.registerFont(TTFont("BodyBold", str(bold)))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="BodyBold", italic="Body", boldItalic="BodyBold")
    m = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
    styles = {
        "title": ParagraphStyle("title", fontName="BodyBold", fontSize=22, leading=26,
                                textColor=colors.black, spaceAfter=12),
        "sub": ParagraphStyle("sub", fontName="Body", fontSize=10, leading=14, spaceAfter=14),
        "h": ParagraphStyle("h", fontName="BodyBold", fontSize=13, leading=17,
                            textColor=colors.black, spaceBefore=11, spaceAfter=6),
        "body": ParagraphStyle("body", fontName="Body", fontSize=10.2, leading=14.1, spaceAfter=7),
        "small": ParagraphStyle("small", fontName="Body", fontSize=8.5, leading=11.5, spaceAfter=6),
        "cell": ParagraphStyle("cell", fontName="Body", fontSize=9, leading=12),
        "headcell": ParagraphStyle("headcell", fontName="BodyBold", fontSize=9, leading=12,
                                   textColor=colors.white),
        "caption": ParagraphStyle("caption", fontName="Body", fontSize=8.4, leading=11,
                                  spaceBefore=4, spaceAfter=8),
    }
    story, md = [], []
    width = A4[0] - 88

    def plain(s):
        s = s.replace("<br/>", "\n\n")
        return html.unescape(re.sub("<[^>]+>", "", s))

    def para(text, style="body"):
        text = text.replace("CO₂", "CO2")
        story.append(Paragraph(text, styles[style]))
        md.append(plain(text) + "\n")

    def heading(text):
        story.append(Paragraph(text, styles["h"]))
        md.append("## " + text + "\n")

    def table(headers, rows, widths):
        cells = [[Paragraph(html.escape(str(c).replace("CO₂", "CO2")), styles["headcell"]) for c in headers]]
        cells += [[Paragraph(html.escape(str(c).replace("CO₂", "CO2")), styles["cell"]) for c in row] for row in rows]
        t = Table(cells, colWidths=[width*w for w in widths], hAlign="LEFT", repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#24475D")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0F4F6")]),
            ("GRID", (0, 0), (-1, -1), .5, colors.HexColor("#D9D9D9")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.extend([t, Spacer(1, 8)])
        md.append("| " + " | ".join(headers) + " |")
        md.append("| " + " | ".join(["---"]*len(headers)) + " |")
        md.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
        md.append("")

    def image(name, caption, image_width=width):
        caption = caption.replace("CO₂", "CO2")
        path = RESULTS / "figures" / name
        img = Image(str(path))
        ratio = img.imageHeight / img.imageWidth
        img.drawWidth, img.drawHeight = image_width, image_width*ratio
        story.append(KeepTogether([img, Paragraph(caption, styles["caption"])]))
        md.extend([f"![{caption}](../analysis/results/figures/{name})", ""])

    def page():
        story.append(PageBreak())

    def percent(value):
        return f"%{100*value:.2f}".replace(".", ",")

    # Page 1: problem, dataset, and preprocessing.
    title = "Çevresel Sensör Verileriyle Oda Doluluk Tespiti"
    story.append(Paragraph(title, styles["title"]))
    md.append("# " + title + "\n")
    para("<b>Makine ve Derin Öğrenme Uygulamaları Proje Raporu</b><br/>"
         "Hazırlayan: Mert Ali Korkmaz<br/>Staj kurumu: Software Persona<br/>Tarih: 9 Kasım 2026", "sub")
    para("Bu çalışmada beş çevresel sensör ölçümünden bir odanın dolu veya boş olduğu tahmin edilmiştir. "
         "PyTorch ile geliştirilen küçük bir yapay sinir ağı, ayrılmış final test verisinde %98,76 doğruluk "
         "ve %97,11 F1 elde etmiştir. Çalışma; veri keşfi, eğitim verisine dayalı ölçekleme, validation ile "
         "model seçimi ve karşılaştırmalı hata analizini içeren bir öğrenme projesidir.")
    heading("1 Problem tanımı")
    para("Odaların kullanım durumunun tahmini, aydınlatma ve iklimlendirme sistemlerinin kullanım ihtiyacına "
         "göre yönetilmesine katkı sağlayabilir. Bu projede sıcaklık, bağıl nem, ışık, CO₂ ve nem oranı "
         "girdi olarak kullanılmıştır. Hedef değişken Occupancy olup 0 boş, 1 dolu odayı göstermektedir. "
         "Amaç, her ölçüm satırını bu iki sınıftan birine atayan ve terminalden kullanılabilen bir model geliştirmektir.")
    heading("2 Veri seti ve değişkenler")
    para("UCI Occupancy Detection veri setinin hazır bölümleri kullanılmıştır [1]. Toplam 20.560 örnekte "
         "sıcaklık (°C), bağıl nem (%), ışık (lux), CO₂ (ppm) ve HumidityRatio (kg su buharı/kg hava) "
         "bulunur. Tarih ve satır kimliği model girdisi değildir. Kaynaktaki doluluk etiketleri, "
         "dakikada bir çekilen zaman damgalı fotoğraflardan elde edilmiştir.")
    table(["Bölüm ve dosya", "Örnek", "Dolu", "Dönem 2015"], [
        ["Eğitim / datatraining.txt", "8.143", "1.729", "4-10 Şubat"],
        ["Validation / datatest.txt", "2.665", "972", "2-4 Şubat"],
        ["Final test / datatest2.txt", "9.752", "2.049", "11-18 Şubat"],
    ], [.42, .13, .13, .32])
    heading("3 Veri ön işleme")
    para("Beş özellikte eksik veya sonlu olmayan değer görülmemiştir; kümeler içinde yinelenen zaman "
         "damgası ve kümeler arasında ortak zaman damgası yoktur. Sayısal girdiler için encoding gerekmez. "
         "Standartlaştırma (x - ortalama) / standart sapma ile uygulanmış; istatistikler yalnızca eğitim "
         "verisinden hesaplanmıştır. Standart sapmada correction=0 kullanılmış, sıfır değerler 1 ile "
         "değiştirilmiştir. Validation, test ve tahmin aşamaları aynı eğitim istatistiklerini kullanır.")
    para("Eğitimde 1,5 IQR kuralı ışıkta 15, CO₂’de 1.063 ve nem oranında 118 uç değer işaretlemiştir. "
         "Bunlar ölçüm hatası olarak doğrulanmadığından veri silinmemiştir; dolulukla ilişkili yüksek "
         "CO₂ değerlerini otomatik silmek bilgi kaybına neden olabilir.", "small")

    # Page 2: three interpretable EDA views.
    page()
    heading("4 Veri keşfi ve görsel bulgular")
    para("İlk analiz sınıf dengesini, ikinci ve üçüncü analizler sensörlerle doluluk arasındaki ilişkiyi "
         "inceler. Sensör dağılımı ve zaman grafiği yalnızca eğitim verisinden üretilmiştir.")
    image("eda_distribution.png", "Şekil 1  Sınıf oranları ve eğitim verisinde doluluk durumuna göre ışık dağılımı", image_width=width-25)
    para("<b>A Sınıf dağılımı.</b> Dolu örnek oranı eğitimde %21,23, validation’da %36,47 ve testte %21,01’dir. "
         "Bu dengesizlik nedeniyle yalnızca accuracy yeterli değildir: testte sürekli boş tahmini bile "
         "%78,99 doğruluk verirken dolu sınıfının recall ve F1 değerleri sıfırdır.")
    para("<b>B Işık dağılımı.</b> Eğitimde ışık medyanı boş odada 0 lux, dolu odada 454 lux’tür. "
         "Bu ayrım ışığın yararlı bir girdi olduğunu düşündürür. Dağılımların örtüşmesi ve uç değerler, "
         "ışığın tek başına hatasız bir doluluk göstergesi olmadığını gösterir.")
    image("eda_timeline.png", "Şekil 2  Eğitim döneminde CO₂ ölçümleri ve gerçek doluluk zamanları", image_width=width-25)
    para("<b>C Zaman içindeki değişim.</b> Eğitimde CO₂ medyanı boş odada 446 ppm, dolu odada 944 ppm’dir. "
         "Grafikte doluluk dönemleriyle birlikte yükselişler görülür; geçişler anlık ve kusursuz değildir. "
         "Havalandırma ve sensör tepkisi gibi etkiler olası açıklamalardır, bu veriyle nedensellik kanıtlanmaz.")
    para("Veri zaman damgalı olsa da mevcut model geçmiş ölçümleri bir dizi olarak kullanmaz. "
         "Bu çalışma, her satırın bağımsız değerlendirildiği bir ikili sınıflandırma uygulamasıdır.", "small")

    # Page 3: methodology and baseline, with no model tuning on the test data.
    page()
    heading("5 Model seçimi ve mimari")
    para("Beş sayısal girdi arasındaki doğrusal olmayan ilişkileri öğrenebilmek için çok katmanlı "
         "algılayıcı (MLP) seçilmiştir. Küçük yapı, PyTorch ile öğrenme sürecinde katmanların ve "
         "gradyan güncellemelerinin anlaşılmasını kolaylaştırır. Bu seçim tüm yöntemlere üstünlük "
         "iddiası taşımaz; daha basit bir yöntemle karşılaştırma ayrıca yapılmıştır.")
    table(["Katman", "Boyut", "Aktivasyon", "Parametre"], [
        ["Gizli katman 1", "5 → 16", "ReLU", "96"],
        ["Gizli katman 2", "16 → 8", "ReLU", "136"],
        ["Çıkış katmanı", "8 → 1", "Ham logit", "9"],
        ["Toplam", "", "", "241"],
    ], [.36, .2, .25, .19])
    para("ReLU doğrusal olmayan dönüşüm sağlar. Kayıp fonksiyonu BCEWithLogitsLoss olduğundan "
         "forward çıktısına sigmoid eklenmemiştir. Değerlendirmede sigmoid ile skor elde edilir; "
         "skor ≥ 0,5 ise dolu, aksi hâlde boş tahmin edilir. Skorların kalibrasyonu ayrıca doğrulanmamıştır.")
    heading("6 Eğitim ve validation yaklaşımı")
    table(["Ayar", "Değer", "Ayar", "Değer"], [
        ["Optimizer", "Adam", "Öğrenme oranı", "0,001"],
        ["Batch size", "32", "Üst epoch sınırı", "100"],
        ["Patience", "5", "PyTorch seed", "42"],
    ], [.25, .25, .25, .25])
    para("Her epoch’ta eğitim batch’leri karıştırılır; zero_grad, forward, backward ve optimizer.step "
         "sırasıyla uygulanır. Ardından eval modunda, gradyan hesaplanmadan validation kaybı ölçülür. "
         "Kayıp önceki en iyi değerden büyük değilse ağırlıkların bağımsız kopyası saklanır; aksi durumda "
         "sayaç artar. Beş ardışık iyileşmeme durumunda eğitim durur; en iyi ağırlıklar geri yüklenerek kaydedilir.")
    para("Mevcut checkpoint değerlendirilmiş, MLP yeniden eğitilmemiştir. Seçilen epoch ve epoch bazlı "
         "kayıp geçmişi checkpoint’te tutulmadığından eğitim eğrisi üretilmemiştir. Seed, farklı paket "
         "sürümleri ve donanımlarda aynı sonucu tek başına garanti etmez.")
    heading("7 Basit modelle karşılaştırma yöntemi")
    para("Logistic Regression, doğrusal karar sınırına sahip bir temel model olarak kullanılmıştır [2]. "
         "StandardScaler ve model yalnızca eğitim bölümünde fit edilmiştir. Ayarlar test sonucu "
         "görülmeden C=1, solver=lbfgs, max_iter=1000 ve random_state=42 olarak sabitlenmiş; model "
         "12 iterasyonda yakınsamıştır. Her iki modelde aynı beş özellik ve 0,5 eşiği kullanılmıştır. "
         "Bu karşılaştırmada hiperparametre araması veya test verisine göre yeniden model seçimi yapılmamıştır.")
    para("Orijinal validation dönemi eğitimden önce gelmektedir; bu nedenle geleceğe yönelik bir "
         "validation simülasyonu değildir. Final test eğitimden sonraki ayrı dönemdedir. Zaman "
         "damgası çakışmaması olumlu olmakla birlikte, yakın zamanlı ölçümler arasında bağımlılık bulunabilir.", "small")

    # Page 4: quantitative evaluation and interpretable trade-offs.
    page()
    heading("8 Değerlendirme sonuçları")
    rows = []
    for key, label in [("validation", "MLP validation"), ("test", "MLP final test")]:
        v = m["metrics"][key]["mlp"]
        rows.append([label] + [percent(v[k]) for k in ["accuracy", "precision", "recall", "f1"]])
    for key, label in [("logistic_regression", "Logistic final test"), ("always_unoccupied", "Hep boş final test")]:
        v = m["metrics"]["test"][key]
        rows.append([label] + [percent(v[k]) for k in ["accuracy", "precision", "recall", "f1"]])
    table(["Model ve bölüm", "Accuracy", "Precision", "Recall", "F1"], rows, [.32,.17,.17,.17,.17])
    para("MLP kaybı eğitimde 0,0426, validation’da 0,0770 ve final testte 0,0694’tür. "
         "Metriklerde pozitif sınıf dolu odadır. Accuracy tüm doğru tahminlerin payını; precision "
         "dolu tahminlerinin doğruluğunu; recall gerçek dolu örneklerin yakalanma oranını gösterir. "
         "F1, precision ile recall’un harmonik ortalamasıdır.")
    image("confusion_matrix.png", "Şekil 3  Satırlar gerçek sınıfı, sütunlar tahmini gösterir  Düzen TN FP ve FN TP", image_width=300)
    para("MLP, 2.049 dolu örneğin 2.036’sını yakalamış, 13’ünü kaçırmıştır. Boş örneklerin "
         "108’ini dolu tahmin etmiştir. Logistic Regression’da bu sayılar sırasıyla 93 kaçırılan dolu "
         "örnek ve 52 yanlış dolu tahminidir. MLP’nin F1 değeri 0,69 yüzde puan daha yüksek; "
         "Logistic Regression’ın precision değeri daha yüksektir.")
    para("Dolu bir odayı boş saymanın daha maliyetli olduğu bir kullanımda MLP’nin yüksek recall’u "
         "yararlı olabilir. Yanlış dolu tahminlerinin gereksiz enerji kullanımına yol açabileceği bir "
         "uygulamada ise precision da önemlidir. Gerçek işletme maliyetleri ölçülmediğinden tek bir "
         "modelin tüm kullanım senaryoları için daha iyi olduğu söylenemez.")
    para("Sonuçlar aynı ofisin sabit veri bölümlerindeki gözlemlerdir; bağımsız binalara genellenebilirlik "
         "ve modeller arasındaki farkın istatistiksel anlamlılığı bu çalışmada sınanmamıştır.", "small")

    # Page 5: actual error evidence, conclusions, and sources.
    page()
    heading("9 Hata analizi")
    para("Final testte 121 yanlış tahmin vardır. Yanlış pozitif (FP), boş odaya dolu; yanlış negatif "
         "(FN), dolu odaya boş denmesidir. Aşağıdaki değerler hata ve doğru tahmin gruplarının "
         "medyanlarını karşılaştırır; neden-sonuç ilişkisi göstermez.")
    table(["Grup", "Örnek", "Işık medyanı lux", "CO₂ medyanı ppm", "Geçişe yakın"], [
        ["TN doğru boş", "7.595", "0,00", "585,00", "44"],
        ["FP yanlış dolu", "108", "361,25", "862,75", "39"],
        ["FN yanlış boş", "13", "279,33", "603,67", "13"],
        ["TP doğru dolu", "2.036", "447,50", "895,88", "71"],
    ], [.23,.12,.23,.24,.18])
    para("Geçişe yakınlık, test etiketinin değiştiği ilk kaydın zamanından en fazla 120 saniye "
         "uzaklık olarak tanımlanmıştır. FN grubundaki 13 örneğin tamamı bu aralıktadır; FP grubunda "
         "39/108 örnek bu aralıktadır. Bu tanımlama gerçek test etiketlerini kullanan geriye dönük bir "
         "incelemedir; model girdisi veya model ayarlama adımı değildir.")
    image("error_scatter.png", "Şekil 4  Test örneklerinin ışık ve CO₂ uzayında doğru ve yanlış tahminleri", image_width=width)
    para("FP grubunun ışık ve CO₂ medyanlarının doğru boş grubundan yüksek olması, bazı boş "
         "ölçümlerin dolu ölçümlere benzemesiyle uyumludur. Örneğin kronolojik ilk FP, 11 Şubat "
         "15:23:59’da 436,5 lux ve 1.047 ppm ile %96,58 doluluk skoru üretmiştir. İlk FN ise aynı "
         "gün 18:23:59’da 279,33 lux ve 585 ppm ile %28,48 skor üretmiştir. Sensör gecikmesi veya "
         "odadan çıkıldıktan sonra yüksek kalan değerler olası açıklamalardır; doğrulanmış nedenler değildir.")
    heading("10 Sonuç ve geliştirme önerileri")
    para("Küçük MLP, bu veri bölümünde yüksek recall ve F1 sağlamış; daha basit Logistic Regression "
         "da güçlü bir temel model olmuştur. Tek ofis ve sınırlı dönem, sınıf oranı değişimi ve geçiş "
         "hataları başlıca sınırlamalardır. Gelecekte farklı oda ve dönemlerden veri toplanmalı, zaman "
         "sırasını koruyan validation uygulanmalı ve geçmiş sensör ölçümlerini kullanan modeller "
         "incelenmelidir. Eşik ve kalibrasyon çalışmaları validation üzerinde yürütülmeli; geliştirme "
         "sonrasında kullanılmak üzere yeni bir bağımsız test seti ayrılmalıdır.")
    heading("Kaynaklar ve yeniden üretim")
    para('[1] Candanedo, L. (2016). Occupancy Detection. UCI. '
         '<link href="https://doi.org/10.24432/C5X01N" color="#24475D">doi.org/10.24432/C5X01N</link>. '
         'Veri lisansı CC BY 4.0; veri dosyaları değiştirilmemiştir.<br/>'
         '[2] scikit-learn, LogisticRegression belgeleri: '
         '<link href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html" color="#24475D">scikit-learn.org</link>.<br/>'
         f'Kod, veri, grafikler ve analiz: <link href="{REPO_URL}" color="#24475D">github.com/merttro/smart-room-occupancy</link>.<br/>'
         'Analiz komutu: python analysis/analyze.py. Python 3.11.9; PyTorch 2.14.1+cpu; '
         'NumPy 1.26.4; scikit-learn 1.8.0; Matplotlib 3.10.9.', "small")

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Body", 8)
        canvas.setFillColor(colors.HexColor("#59636B"))
        canvas.drawString(44, 28, "Mert Ali Korkmaz  |  Software Persona  |  Proje raporu")
        canvas.drawRightString(A4[0]-44, 28, f"{doc.page} / 5")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(out / "oda_doluluk_proje_raporu.pdf"), pagesize=A4,
        rightMargin=44, leftMargin=44, topMargin=36, bottomMargin=54,
        title=title, author="Mert Ali Korkmaz", subject="Software Persona staj projesi")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    # Keep Markdown image links valid in both repository report/ and a copied output folder.
    (out / "oda_doluluk_proje_raporu.md").write_text("\n".join(md), encoding="utf-8")
    print("Created:", out / "oda_doluluk_proje_raporu.pdf")


if __name__ == "__main__":
    main()

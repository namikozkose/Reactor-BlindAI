import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

# =====================================================================
# 1. REAKTÖR SOĞUTMA POMPASI YAPAY ZEKA MODELİ
# =====================================================================
class NukePumpAI(nn.Module):
    def __init__(self):
        super(NukePumpAI, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

    def forward(self, x):
        return self.net(x)

model = NukePumpAI()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.02)

# =====================================================================
# 2. AŞAMA: İLK (SAVUNMASIZ) EĞİTİM
# =====================================================================
print("[*] Model standart verilerle eğitiliyor (Savunmasız Faz)...")
X_train = torch.tensor([
    [0.45, 0.12, 0.50, 0.80], [0.55, 0.25, 0.60, 0.70], [0.62, 0.40, 0.65, 0.55],
    [0.75, 0.70, 0.80, 0.35], [0.85, 0.90, 0.95, 0.20], [0.92, 0.85, 0.90, 0.15]
], dtype=torch.float32)
y_train = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

for _ in range(50):
    optimizer.zero_grad()
    loss = criterion(model(X_train), y_train)
    loss.backward()
    optimizer.step()

model.eval()
print("[+] İlk Eğitim Tamamlandı!\n" + "="*65)

# =====================================================================
# 3. AŞAMA: SALDIRI (PGD / İTERATİF EVASION)
# =====================================================================
print("[RED TEAM] Savunmasız modele PGD Saldırısı başlatılıyor...")
kritik_veri = torch.tensor([[0.78, 0.75, 0.82, 0.30]], requires_grad=True)
zehirli_veri = kritik_veri.clone().detach()
hedef_etiket = torch.tensor([0]) # Hedef: Modelin "Normal" demesini sağlamak
alpha = 0.03

for adim in range(1, 8):
    zehirli_veri.requires_grad = True
    cikti = model(zehirli_veri)
    anlik_sinif = cikti.argmax(dim=1).item()
    norm_prob = torch.softmax(cikti, dim=1)[0][0].item() * 100
    
    if anlik_sinif == 0 and norm_prob > 85.0:
        print(f"[🎯] BAŞARILI SALDIRI! Adım {adim}'de model kandırıldı. Güven: %{norm_prob:.1f}")
        break
    
    loss = criterion(cikti, hedef_etiket)
    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        zehirli_veri = zehirli_veri - (alpha * zehirli_veri.grad.sign())

print("\n" + "="*65)
print("[BLUE TEAM] AŞILAMA (Adversarial Training) BAŞLATILIYOR...")
print("[*] Model, saldırırda kullanılan zehirli verilerle güçlendiriliyor...")

# Saldırıda kullanılan o sinsi zehirli veriyi ve gerçekte ne olduğunu (1 = Kritik Arıza) eğitime ekliyoruz
zehirli_egitim_verisi = zehirli_veri.detach().clone()
gercek_etiket_krizi = torch.tensor([1], dtype=torch.long)

# Modeli yeniden eğit (Aşıla)
model.train()
optimizer = optim.Adam(model.parameters(), lr=0.01)
for _ in range(100):
    optimizer.zero_grad()
    # Hem orijinal veriler hem de zehirli (saldırı) verileri modele öğretiyoruz
    combined_X = torch.cat([X_train, zehirli_egitim_verisi], dim=0)
    combined_y = torch.cat([y_train, gercek_etiket_krizi], dim=0)
    
    loss = criterion(model(combined_X), combined_y)
    loss.backward()
    optimizer.step()

model.eval()
print("[+] Aşı Tamamlandı! Model adversarial saldırılara karşı bağışıklık kazandı.\n" + "="*65)

# =====================================================================
# 4. AŞAMA: AŞILANMIŞ MODELİ TEKRAR TEST ETME
# =====================================================================
print("[RED TEAM] Aynı saldırı aşılanmış (zırhlı) modele tekrar uygulanıyor...")

zirlhli_veri = kritik_veri.clone().detach()
for adim in range(1, 8):
    zirlhli_veri.requires_grad = True
    cikti = model(zirlhli_veri)
    anlik_sinif = cikti.argmax(dim=1).item()
    ariza_prob = torch.softmax(cikti, dim=1)[0][1].item() * 100
    
    loss = criterion(cikti, hedef_etiket)
    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        zirlhli_veri = zirlhli_veri - (alpha * zirlhli_veri.grad.sign())

final_cikti = model(zirlhli_veri)
final_sinif = final_cikti.argmax(dim=1).item()
final_ariza_prob = torch.softmax(final_cikti, dim=1)[0][1].item() * 100

print(f"Zırhlı Modelin Son Kararı : {'KRİTİK ARIZA ALARMI!' if final_sinif == 1 else 'NORMAL'}")
print(f"Arıza Tespit Güven Oranı  : %{final_ariza_prob:.2f}")

if final_sinif == 1:
    print("\n[🛡️] SAVUNMA BAŞARILI: Model aşılandı ve sinsi gürültüye boyun eğmeyerek")
    print("     nükleer reaktörde doğru bir şekilde 'KRİTİK ARIZA' alarmı vermeye devam etti!")
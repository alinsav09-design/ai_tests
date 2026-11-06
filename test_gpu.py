import torch
import time

# Ne asigurăm că mutăm calculele pe placa video (cuda)
device = torch.device("cuda")

print(f"🚀 Rulăm calculele pe: {torch.cuda.get_device_name(0)}")

# 1. Creăm două matrice uriașe (10.000 x 10.000) direct în memoria plăcii video (VRAM)
print("⏳ Generăm matricele în VRAM...")
x = torch.rand(10000, 10000, device=device)
y = torch.rand(10000, 10000, device=device)

# 2. Măsurăm timpul de execuție pentru înmulțirea lor pe GPU
start_time = time.time()

# Efectuăm înmulțirea matricială (operația de bază din Deep Learning)
rezultat = torch.matmul(x, y)

# Punct de sincronizare pentru a ne asigura că GPU-ul a terminat calculul înainte de a opri ceasul
torch.cuda.synchronize()

end_time = time.time()
print(f"✅ Înmulțirea a fost finalizată cu succes!")
print(f"⏱️ Timp de execuție pe GPU: {end_time - start_time:.4f} secunde")


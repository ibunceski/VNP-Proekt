# import torch
# import torch.nn as nn
#
# def train_model(model, train_loader, valid_loader,
#                 epochs=50, lr=1e-3, patience=5, device="cpu"):
#     model = model.to(device)
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
#
#     best_val = float("inf")
#     patience_counter = 0
#     best_model = None
#
#     for epoch in range(1, epochs + 1):
#         # --- training ---
#         model.train()
#         total_loss = 0
#         for xb, yb in train_loader:
#             xb, yb = xb.to(device), yb.to(device)
#             optimizer.zero_grad()
#             preds = model(xb)
#             loss = criterion(preds, yb)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item() * xb.size(0)
#
#         avg_train = total_loss / len(train_loader.dataset)
#
#         # --- validation ---
#         model.eval()
#         total_val = 0
#         with torch.no_grad():
#             for xb, yb in valid_loader:
#                 xb, yb = xb.to(device), yb.to(device)
#                 preds = model(xb)
#                 loss = criterion(preds, yb)
#                 total_val += loss.item() * xb.size(0)
#         avg_val = total_val / len(valid_loader.dataset)
#
#         scheduler.step(avg_val)
#         print(f"Epoch {epoch:03d} | Train {avg_train:.6f} | Val {avg_val:.6f}")
#
#         if avg_val < best_val:
#             best_val = avg_val
#             best_model = model.state_dict()
#             patience_counter = 0
#         else:
#             patience_counter += 1
#             if patience_counter >= patience:
#                 print("Early stopping triggered")
#                 break
#
#     if best_model:
#         model.load_state_dict(best_model)
#     return model

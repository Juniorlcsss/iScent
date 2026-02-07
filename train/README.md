g++ -std=c++17 -O2 -Wall -I./train -I./src/models train/ml_trainer.cpp train/csv_loader.cpp src/models/dt.cpp src/models/knn.cpp src/models/rf.cpp -o ml_trainer.exe

TOML Configuration File Structure
├── sys_mode/
│   ├── debug
│   ├── cover_exist
│   ├── train_tasks_per_thread
│   └── predict_tasks_per_thread
├── folder/
│   ├── model
│   ├── eval
│   └── pred
├── dataset/
│   ├── TVARs[]/
│   │   └── (Item Structure)
│   │       ├── name
│   │       ├── path
│   │       └── variable
│   ├── CVARs[]/
│   │   └── (Item Structure)
│   │       ├── name
│   │       ├── path
│   │       └── variable
│   └── mask/
│       ├── path
│       ├── variable
│       └── total
├── model/
│   ├── hidden_dim
│   ├── n_layers
│   ├── batch_size
│   ├── seq_length
│   └── window_size
├── train/
│   ├── start_date
│   ├── end_date
│   ├── train_years
│   ├── lr
│   ├── n_epochs
│   ├── patience
│   └── verbose_epoch
└── predict/
    ├── start_date
    └── end_date
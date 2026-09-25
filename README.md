## .reg to .txt conversion script

```
python3 common/ds9_regions_to_catalogue.py \
        data/curated/saods9/name_of_galaxy.reg \
        -o  data/curated/catalogue/name_of_galaxy.txt
```
    
## Layout

```

bubbleDetection/
├── README.md                  
│
├── data/                     
│   ├── raw/                   
│       └── catalogue/                   
│       └── fits/                   
│           └── moreGalaxies/                   
│   └── curated/    
│       └── catalogue/                   
│       └── saods9/                   
│   └── processed_seg/
│       └── ...                   
│   └── processed_det/
│
├── common/                    
│   ├── ds9_regions_to_catalogue.py
│   ├── mergeCatalogues.py
│   └── whereismybubble.py
│
├── segmentation/              
│   ├── README.md              
│   ├── notebooks/             
│   ├── scripts/               
│   ├── data/                  
│   └── outputs/               
│
└── detection/                 
    ├── README.md
    ├── notebooks/            
    ├── scripts/
    ├── data/                
    └── outputs/             
    
```
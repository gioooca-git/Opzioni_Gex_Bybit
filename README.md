3 filese that use Bybit as exchange:  


1


Bybit_0DTE_e_ALL_scaricoPipeLine_creazioneDF_CalcoloGex_1.2.py 

show gex, call wall, put wall, spot, gamma flip into .\data_bybit_btc_options 

  save in .\data_bybit_btc_options the following files (images,csv, jsson): 
  
    0DTE_Grafico......png   GEX,gamma filp, ecc.. image for 0DTE options chain 
    
    ALL_Grafico.......png   GEX,gamma filp, ecc.. image for ALL aoptions chain 
    
    0DTE_dati..........csv  csv with datas for 0DTE options chain 
    
    ALL_dati..........csv   csv with datas for ALL options chain 
    
    0DTE_metadati..........json  json with datas for 0DTE options chain 
    
    ALL_metadati..........json   json with datas for ALL options chain 
    
2


CalcoloGEXTemporizzatoDa_BYBIT_2.1.py 

  show daily gex,gammaflip, etc... values and save (images,csv, jsson) in .\data_bybit_60s\yyy-mm-dd the following files: 
  
     GEX_dashboard_ALL_BTC_yyy-mm-dd.png    image for ALL options chains gamma filip,gex, temporal variations 
     
     GEX_dashboard_0DTE_BTC_yyy-mm-dd.png   image for 0DTE options chains gamma filip,gex, temporal variations 
     
     GEX_0DTE_ALL_BTC_yyy-mm-dd.csv         datas for 0DTE/ALL options chains gamma filip,gex, temporal variations 
     
     GEX_dashboard_ALL_BTC_yyy-mm-ddpng.meta.json    metadatas for ALL options chains gamma filip,gex, temporal variations 
     
     GEX_dashboard_0DTE_BTC_yyy-mm-dd.png.meta.json  metadatas for 0DTE options chains gamma filip,gex, temporal variations 
     

      where yyyy = yaer  mm=month dd=day 
      
3


dashboard_live1.1.py  to show realtime datas changing in browser for last day (GEX,gamma flip, ecc...) 


this script use streamlit and should be executed with: 

streamlit run dashboard_livex.y.py  



     
  

È possibile testare la generazione di file con Jinja eseguendo il file source.py che a sua volta eseguirà il file jinja.py che genererà i file necessari per l'interfaccia di ipopt in c++ a partire dai template inseriti nella cartella template.
Il file source.py è il file all'interno del quale si inseriscono e modificano i parametri del problema. 
All'interno dei template sostituiti alcuni parametri costanti come le dimensioni del problema, i boundaries dalle variabili e dei constrains.
In quest'esempio viene utilizzato un problema standard di test con 5 variabili e 3 constrain.
Per testare interamente il codice e verificare il corretto funzionamento di ipopt, è necessario modificare, una volta generato, il file CMakeLists.txt inserendo il path corretto alla cartella di installazione di ipopt con Homebrew.
In seguito è sufficiente muoversi nella cartella build eseguendo in serie i comandi "cmake .." e "make".
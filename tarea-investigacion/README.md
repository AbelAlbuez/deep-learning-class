
# Ejecución de los resultados

Se debe crear un entorno virutal de python mediante uv con python 3.12 para que no exista conflicto de dependencias 

```
    uv python install 3.12
    uv venv --python 3.12
```  

Una vez creado se debe activar 

```
    source .venv/bin/activate
```  
La carpeta contiene el archivo pyproject.toml que contiene las dependencias necesarias. Se debe ejecutar el comando.

```
    uv sync
```  

Antes de lanzar jupyter se debe exportar el entorno. Esto se hace para que no se tengan que instalar nuevamente las librearías en jupyter y solo haya que importarlas. Para ello se debe instalar ipykernel

```
    uv pip install ipykernel
``` 

Ahora si se exporta el entorno virutal para que lo reconozca jupyter

```
    python -m ipykernel install --user --name=entorno-uv-312 --display-name="Python 3.12 (Mi Entorno UV)"
```

Despues de eso se lanza una instancia de jupyterlab usando el siguiente comando

```
    uvx --with jupyter jupyter notebook
```

Esto automáticamente abre el jupyterlab en el navegador donde solo hay que cambiar el kernel por el que creamos seleccionado: "Mi entorno uv".

El archivo que contiene el código fuente con los resultados obtenidos se llama 

```
    fine-tuning.ipynb
```
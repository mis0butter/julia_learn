begin 
    using Colors, ColorVectorSpace, ImageShow, FileIO, ImageIO 
    using PlutoUI 
    using HypertextLiteral 
end 

## use Images.jl to load an image file 

# Step 1: specify URL 
url = "https://user-images.githubusercontent.com/6933510/107239146-dcc3fd00-6a28-11eb-8c7b-41aaf6618935.png" 

# Step 2: download 
philip_filename = download(url) # download to a local file. The filename is returned 

# Step 3: load file 
philip = load(philip_filename)

## Exercise: change URL 
## Exercise: download a file already on your computer 
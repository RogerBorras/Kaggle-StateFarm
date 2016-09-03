################################################
#############Preprocessing the data#############
################################################

# Resize images and convert to grayscale

rm(list=ls())
require(EBImage)

# Set wd where images are located
setwd("C://dogs_images")
# Set d where to save images
save_in <- "C://dogs_images_resized"
# Load images names
images <- list.files()
# Set width
w <- 28
# Set height
h <- 28

# Main loop resize images and set them to greyscale
for(i in 1:length(images))
{
    # Try-catch is necessary since some images
    # may not work.
    result <- tryCatch({
    # Image name
    imgname <- images[i]
    # Read image
    img <- readImage(imgname)
    # Resize image 28x28
    img_resized <- resize(img, w = w, h = h)
    # Set to grayscale
    grayimg <- channel(img_resized,"gray")
    # Path to file
    path <- paste(save_in, imgname, sep = "")
    # Save image
    writeImage(grayimg, path, quality = 70)
    # Print status
    print(paste("Done",i,sep = " "))},
    # Error function
    error = function(e){print(e)})
}


###################################################
#############Stored in a proper format#############
###################################################

# Generate a train-test dataset

# Clean environment and load required packages
rm(list=ls())
require(EBImage)

# Set wd where resized greyscale images are located
setwd("C://dogs_resized")

# Out file
out_file <- "C://dogs_28.csv"

# List images in path
images <- list.files()

# Set up df
df <- data.frame()

# Set image size. In this case 28x28
img_size <- 28*28

# Set label
label <- 1

# Main loop. Loop over each image
for(i in 1:length(images))
{
    # Read image
    img <- readImage(images[i])
    # Get the image as a matrix
    img_matrix <- img@.Data
    # Coerce to a vector
    img_vector <- as.vector(t(img_matrix))
    # Add label
    vec <- c(label, img_vector)
    # Bind rows
    df <- rbind(df,vec)
    # Print status info
    print(paste("Done ", i, sep = ""))
}

# Set names
names(df) <- c("label", paste("pixel", c(1:img_size)))

# Write out dataset
write.csv(df, out_file, row.names = FALSE)

#-------------------------------------------------------------------------------
# Test and train split and shuffle

# Load datasets
plants <- read.csv("plants_28.csv")
dogs <- read.csv("dogs_28.csv")

# Bind rows in a single dataset
new <- rbind(plants, dogs)

# Shuffle new dataset
shuffled <- new[sample(1:1512),]

# Train-test split
train_28 <- shuffled[1:1200,]
test_28 <- shuffled[1201:1512,]

# Save train-test datasets
write.csv(train_28, "train_28.csv",row.names = FALSE)
write.csv(test_28, "test_28.csv",row.names = FALSE)


########################################################
#############Building and testing the model#############
########################################################

rm(list=ls())

# Load MXNet
require(mxnet)

# Train test datasets
train <- read.csv("train_28.csv")
test <- read.csv("test_28.csv")

# Fix train and test datasets
train <- data.matrix(train)
train_x <- t(train[,-1])
train_y <- train[,1]
train_array <- train_x
dim(train_array) <- c(28, 28, 1, ncol(train_x))

test__ <- data.matrix(test)
test_x <- t(test[,-1])
test_y <- test[,1]
test_array <- test_x
dim(test_array) <- c(28, 28, 1, ncol(test_x))

# Model
data <- mx.symbol.Variable('data')
# 1st convolutional layer 5x5 kernel and 20 filters.
conv_1 <- mx.symbol.Convolution(data= data, kernel = c(5,5), num_filter = 20)
tanh_1 <- mx.symbol.Activation(data= conv_1, act_type = "tanh")
pool_1 <- mx.symbol.Pooling(data = tanh_1, pool_type = "max", kernel = c(2,2), stride = c(2,2))
# 2nd convolutional layer 5x5 kernel and 50 filters.
conv_2 <- mx.symbol.Convolution(data = pool_1, kernel = c(5,5), num_filter = 50)
tanh_2 <- mx.symbol.Activation(data = conv_2, act_type = "tanh")
pool_2 <- mx.symbol.Pooling(data = tanh_2, pool_type = "max", kernel = c(2,2), stride = c(2,2))
# 1st fully connected layer
flat <- mx.symbol.Flatten(data = pool_2)
fcl_1 <- mx.symbol.FullyConnected(data = flat, num_hidden = 500)
tanh_3 <- mx.symbol.Activation(data = fcl_1, act_type = "tanh")
# 2nd fully connected layer
fcl_2 <- mx.symbol.FullyConnected(data = tanh_3, num_hidden = 2)
# Output
NN_model <- mx.symbol.SoftmaxOutput(data = fcl_2)

# Set seed for reproducibility
mx.set.seed(100)

# Device used. Sadly not the GPU :-(
device <- mx.cpu()

# Train on 1200 samples
model <- mx.model.FeedForward.create(NN_model, X = train_array, y = train_y,
                                     ctx = device,
                                     num.round = 30,
                                     array.batch.size = 100,
                                     learning.rate = 0.05,
                                     momentum = 0.9,
                                     wd = 0.00001,
                                     eval.metric = mx.metric.accuracy,
                                     epoch.end.callback = mx.callback.log.train.metric(100))

# Test on 312 samples
predict_probs <- predict(model, test_array)
predicted_labels <- max.col(t(predict_probs)) - 1
table(test__[,1], predicted_labels)
sum(diag(table(test__[,1], predicted_labels)))/312

##############################################
# Output
##############################################
#   predicted_labels
#      0   1
#  0  83  47
#  1  34 149
#
#
# [1] 0.7412141
#


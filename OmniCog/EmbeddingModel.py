{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "41acfa9c",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Downloading file 'captions_train2014.json' from 'https://github.com/rsokl/cog_data/releases/download/language-files/captions_train2014.json' to '/Users/23amritap/Library/Caches/cog_data'.\n"
     ]
    }
   ],
   "source": [
    "from cogworks_data.language import get_data_path\n",
    "\n",
    "from pathlib import Path\n",
    "import json\n",
    "\n",
    "# load COCO metadata\n",
    "filename = get_data_path(\"captions_train2014.json\")\n",
    "with Path(filename).open() as f:\n",
    "    coco_data = json.load(f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "48b7a5dc",
   "metadata": {},
   "outputs": [],
   "source": [
    "from gensim.models import KeyedVectors\n",
    "filename = \"glove.6B.200d.txt.w2v\"\n",
    "\n",
    "# this takes a while to load -- keep this in mind when designing your capstone project\n",
    "glove = KeyedVectors.load_word2vec_format(get_data_path(filename), binary=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "8a97bb34",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Downloading file 'resnet18_features.pkl' from 'https://github.com/rsokl/cog_data/releases/download/language-files/resnet18_features.pkl' to '/Users/23amritap/Library/Caches/cog_data'.\n"
     ]
    }
   ],
   "source": [
    "import pickle\n",
    "with Path(get_data_path('resnet18_features.pkl')).open('rb') as f:\n",
    "    resnet18_features = pickle.load(f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "da6180be",
   "metadata": {},
   "outputs": [],
   "source": [
    "from mynn.layers.dense import dense\n",
    "from mygrad.nnet.initializers import glorot_normal\n",
    "import numpy as np\n",
    "\n",
    "class Model():\n",
    "    \"\"\"\n",
    "    Model that creates semantic space embeddings using the original image vectors\n",
    "    \"\"\"\n",
    "    def __init__(self, input_dim, output_dim):\n",
    "        \"\"\" \n",
    "        Initializes the layer\n",
    "        \n",
    "        Parameters\n",
    "        ----------\n",
    "        dim_input: int \n",
    "            The original image descriptor vector dimension\n",
    "        \n",
    "        dim_output: int\n",
    "            The final image embedding dimension\n",
    "        \"\"\"\n",
    "        self.dense_layer = dense(input_dim, output_dim, weight_initializer=glorot_normal, bias=False)\n",
    "        \n",
    "    def __call__(self, x):\n",
    "        \"\"\"\n",
    "        Does one forward pass of the network\n",
    "        \n",
    "        Parameters\n",
    "        ----------\n",
    "        x: Union[numpy.ndarray, mygrad.Tensor], shape=(training data length, input_dim)\n",
    "            The training data of image descriptor vectors\n",
    "        \n",
    "        Returns\n",
    "        -------\n",
    "        mygrad.Tensor, shape=(training data length, output_dim)\n",
    "            The normalized image embeddings predicted by the model\n",
    "        \"\"\"\n",
    "        embeddings = self.dense_layer(x)\n",
    "        return embeddings / np.linalg.norm(embeddings)\n",
    "        \n",
    "    @property\n",
    "    def parameters(self):\n",
    "        \"\"\"\n",
    "        Gets the model's parameters\n",
    "        \n",
    "        Returns\n",
    "        -------\n",
    "        Tuple[Tensor, ...]\n",
    "            The weights of the model\n",
    "        \"\"\"\n",
    "        return self.dense_layer.parameters"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "4314390f",
   "metadata": {},
   "outputs": [],
   "source": [
    "def load_weights():\n",
    "    \"\"\" \n",
    "        Loads the weights from the numpy array file\n",
    "        \n",
    "        Returns\n",
    "        -------\n",
    "        The array of weights\n",
    "        \"\"\"\n",
    "    return np.load('WeightsArray.npy')\n",
    "\n",
    "def save_weights(arr: np.array):\n",
    "    \"\"\" \n",
    "        Stores the weights to the numpy array file\n",
    "        \n",
    "        Parameters\n",
    "        ----------\n",
    "        arr: np.array\n",
    "            The array of weights\n",
    "        \"\"\"\n",
    "    np.save('WeightsArray.npy', arr)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "7feb7214",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "from noggin import create_plot\n",
    "plotter, fig, ax = create_plot(metrics=[\"loss\", \"accuracy\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "c37838ad",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'extract_triples' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Input \u001b[0;32mIn [31]\u001b[0m, in \u001b[0;36m<cell line: 9>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      6\u001b[0m epochs \u001b[38;5;241m=\u001b[39m \u001b[38;5;241m10\u001b[39m\n\u001b[1;32m      7\u001b[0m batch_size \u001b[38;5;241m=\u001b[39m \u001b[38;5;241m32\u001b[39m\n\u001b[0;32m----> 9\u001b[0m training_data, testing_data \u001b[38;5;241m=\u001b[39m \u001b[43mextract_triples\u001b[49m(path) \u001b[38;5;66;03m# what is path\u001b[39;00m\n\u001b[1;32m     11\u001b[0m \u001b[38;5;28;01mfor\u001b[39;00m iteration \u001b[38;5;129;01min\u001b[39;00m \u001b[38;5;28mrange\u001b[39m(\u001b[38;5;241m0\u001b[39m, epochs):\n\u001b[1;32m     12\u001b[0m     indices \u001b[38;5;241m=\u001b[39m np\u001b[38;5;241m.\u001b[39marange(\u001b[38;5;28mlen\u001b[39m(training_data))\n",
      "\u001b[0;31mNameError\u001b[0m: name 'extract_triples' is not defined"
     ]
    }
   ],
   "source": [
    "from mynn.optimizers.sgd import SGD\n",
    "\n",
    "model = Model(512, 200)\n",
    "optimizer = SGD(model.parameters, learning_rate = 1e-3, momentum=0.9)\n",
    "\n",
    "epochs = 10\n",
    "batch_size = 32\n",
    "\n",
    "training_data, testing_data = extract_triples(path) # what is path\n",
    "\n",
    "for iteration in range(0, epochs):\n",
    "    indices = np.arange(len(training_data))\n",
    "    np.random.shuffle(indices)\n",
    "    \n",
    "    for batch_count in range(0, len(training_data)//batch_size):\n",
    "        batch_indices = indices[batch_count * batch_size : (batch_count+1) * batch_size]\n",
    "        batch = training_data[batch_indices]\n",
    "        \n",
    "        true_embedding = model([resnet18_features(true_vec) for (caption, true_vec, confuse_vec) in batch]) \n",
    "        confuser_embedding = model([resnet18_features(confuse_vec) for (caption, true_vec, confuse_vec) in batch])\n",
    "        \n",
    "        caption_embedding = [glove(caption) for (caption, true_vec, confuse_vec) in batch]\n",
    "        \n",
    "        loss = loss(true_embedding, caption_embedding, confuser_embedding)\n",
    "        accuracy = accuracy(loss)\n",
    "        \n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "        \n",
    "        plotter.set_train_batch({\"loss\":loss.item(), \"accuracy\":accuracy}, batch_size=batch_size, plot=False)\n",
    "\n",
    "    test_indices = np.arange(len(testing_data))\n",
    "    np.random.shuffle(test_indices)\n",
    "        \n",
    "    for batch_count in range(0, len(testing_data)//batch_size):\n",
    "        test_batch_indices = test_indices[batch_count * batch_size : (batch_count+1) * batch_size]\n",
    "        test_batch = training_data[test_batch_indices]\n",
    "        \n",
    "        true_embedding = model([resnet18_features(true_vec) for (caption, true_vec, confuse_vec) in test_batch]) \n",
    "        confuser_embedding = model([resnet18_features(confuse_vec) for (caption, true_vec, confuse_vec) in test_batch])\n",
    "        \n",
    "        caption_embedding = [glove(caption) for (caption, true_vec, confuse_vec) in test_batch]\n",
    "        \n",
    "        test_loss = loss(true_embedding, caption_embedding, confuser_embedding)\n",
    "        test_accuracy = accuracy(test_loss)\n",
    "        \n",
    "        plotter.set_test_batch({\"loss\":test_loss.item(), \"accuracy\":test_accuracy}, batch_size=batch_size, plot=False)\n",
    "    \n",
    "        \n",
    "    plotter.set_train_epoch()\n",
    "    plotter.set_test_epoch()\n",
    "        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "54c4079f",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

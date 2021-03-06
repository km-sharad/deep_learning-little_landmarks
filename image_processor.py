from PIL import Image

def main():
	im = Image.open("DogWalkingLittleLandmarks/prototypical/dog-walking40.jpg")
	file_object  = open("DogWalkingLittleLandmarks/prototypical/dog-walking40.labl", "r") 
	labl = file_object.read()
	print(labl)

	labl_lst = labl.split("|")

	labl_count = int(labl_lst[2])

	print(labl_count)

	j = 3
	coord_tupl_lst = []
	labl_tupl_lst = []
	for i in range (0,labl_count):
		#print((labl_lst[j],labl_lst[j+1],labl_lst[j+2],labl_lst[j+3]))
		coord_tupl_lst.append((labl_lst[j],labl_lst[j+1],labl_lst[j+2],labl_lst[j+3]))
		j=j+4

	print(coord_tupl_lst)

	for i in range (0,labl_count):
		labl_tupl_lst.append(labl_lst[j])
		j = j+1

	print(labl_tupl_lst)	

	holding_leash_idx = labl_tupl_lst.index('holding-leash')
	print(coord_tupl_lst[holding_leash_idx])


	crop_box = tuple(map(int, coord_tupl_lst[holding_leash_idx]))
	print(crop_box)

	crop_coord = (crop_box[0], crop_box[1], crop_box[0] + crop_box[2], crop_box[1] + crop_box[3])
	print(crop_coord)

	cropped_im = im.crop(crop_coord)
	print(cropped_im.size)
	cropped_im.show()

if __name__ == "__main__":
    main()
def Set_equal_dimensions_2_IQ(vect1,vect2):
	##this fuction adds 0 to the shorter array in order to 
	## have same dimension in both arrays
	len1 = len(vect1)
	len2 = len(vect2)
	if(len1>len2):
		for i in range(len1-len2):
			vect2.append(0)
	if(len2>len1):
		for i in range(len2-len1):
			vect2.append(0)

	return(vect1,vect2)


def operating_array(vect1,vect2,operation):
	if(operation=="dot product"):
		dotProduct=[]
		for i in range(int(len(vect1))):
			dotProduct.append(vect1[i]*vect2[i])
		return dotProduct
	elif(operation=="substraction"):
		Substraction=[]
		for i in range(int(len(vect1))):
			Substraction.append(vect1[i]-vect2[i])
		return Substraction
	elif(operation=="sumation"):
		Sum=[]
		for i in range(int(len(vect1))):
			sum.append(vect1[i]+vect2[i])
		return Sum
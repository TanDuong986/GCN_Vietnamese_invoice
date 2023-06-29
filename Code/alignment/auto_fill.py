from __init__ import *
'''nhiệm vụ của file này còn phải tạo ra vùng bounding để bỏ những thông tin không nằm trong vùng chứa hóa đơn tổng
    bên cạnh đó khảo sát để bỏ những hóa đơn không có một cạnh nào thẳng ,không tuân theo giới hạn để duỗi thằng
    đồng thời bỏ các hóa đơn có độ tin cậy đánh nhãn không quá cao.
''' 

'''
padding một chút vùng đen xung quang hóa đơn để bao trọn được hết hóa đơn => sinh ra contour area'''
def puring(img,rgb):
    ''''
    fill all the area in invoice with same color of mask (1)
    input: RGB image
    ____
    output : mask of image after pured
    '''
    img_cp = img.copy()
    gray = cv2.cvtColor(img_cp,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    cny = cv2.Canny(thresh,10,200)
    ctn, hie = cv2.findContours(cny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for ct in ctn:
        print(f'area : {cv2.contourArea(ct)}')
        print(f'lenght : {cv2.arcLength(ct,False)}')
    # threshold_area = 100  # Set your desired area threshold
    # sorted_contours = sorted(ctn, key=cv2.contourArea, reverse=True)
    # filtered_contours = [contour for contour in sorted_contours if cv2.contourArea(contour) > threshold_area]
    # cv2.drawContours(img_cp,ctn,-1,(0,0,255),thickness=2)
    # mask = np.squeeze(np.array([thresh >= 10]).astype(np.uint8)) # h,w
    # rgb *= mask[:, :, np.newaxis]

    
    


if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',type=str
                        ,default='/home/dtan/Documents/GCN/GCN_Vietnam/Code/U2Net/output/mcocr_private_145120pxmgi.jpg',
                        help="source image")
    otp = parser.parse_args()

    anh = cv2.imread(otp.source)
    name_image = os.path.splitext(os.path.basename(otp.source))[0]
    rgb = cv2.imread(os.path.join('/home/dtan/Documents/GCN/GCN_Vietnam/Vietnam_invoice_data/mcocr2021_raw/test/test_images',name_image+".jpg"))
    puring(anh,rgb)


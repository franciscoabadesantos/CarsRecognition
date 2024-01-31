__kernel void gray_implementation(
    __read_only image2d_t src,
    __write_only image2d_t dst)
{
    const sampler_t samp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    // receber o pixel
    int2 coord = (int2)(get_global_id(0), get_global_id(1));

    uint4 pixel = read_imageui(src, samp, coord);

    // transformar o pixel em cinzentos
    float gray = 0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z;
    uint4 gray_pixel = (uint4)(gray, gray, gray, 255);

    write_imageui(dst, coord, gray_pixel);
}



__kernel void sobel_implementation(
    __read_only image2d_t src,
    __write_only image2d_t dst)
{
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    int2 coord = (int2)(get_global_id(0), get_global_id(1));

    uint4 Pixel00 = read_imageui(src, sampler, (int2)(coord.x - 1, coord.y - 1));
    uint4 Pixel01 = read_imageui(src, sampler, (int2)(coord.x, coord.y - 1));
	uint4 Pixel02 = read_imageui(src, sampler, (int2)(coord.x + 1, coord.y - 1));

	uint4 Pixel10 = read_imageui(src, sampler, (int2)(coord.x - 1, coord.y));
	uint4 Pixel12 = read_imageui(src, sampler, (int2)(coord.x + 1, coord.y));

	uint4 Pixel20 = read_imageui(src, sampler, (int2)(coord.x - 1, coord.y + 1));
	uint4 Pixel21 = read_imageui(src, sampler, (int2)(coord.x, coord.y + 1));
	uint4 Pixel22 = read_imageui(src, sampler, (int2)(coord.x + 1, coord.y + 1));

    uint4 Gx = Pixel00 + (2 * Pixel10) + Pixel20 - Pixel02 - (2 * Pixel12) - Pixel22;
    uint4 Gy = Pixel00 + (2 * Pixel01) + Pixel02 - Pixel20 - (2 * Pixel21) - Pixel22;

    uint4 G = (uint4)(0, 0, 0, Pixel00.w);
    G.x = sqrt((float)(Gx.x * Gx.x + Gy.x * Gy.x)); // B
    G.y = sqrt((float)(Gx.y * Gx.y + Gy.y * Gy.y)); // G
	G.z = sqrt((float)(Gx.z * Gx.z + Gy.z * Gy.z)); // R

	double diff = abs((int)(G.z-G.y)) + abs((int)(G.z-G.x)) + abs((int)(G.y-G.x));
    double average = (G.x+G.y+G.z)/3;

    if(diff>10 && average>50){
        write_imageui( dst, (int2)(coord.x,coord.y) , (uint4)(255,255,255,0));
    }
    else
        write_imageui( dst, (int2)(coord.x,coord.y) , (uint4)(0,0,0,0));
}


__kernel void hough_implementation(
    __read_only image2d_t image,
    __global int* HS1,
    __global int* HS2,
    int width,
    int height,
    int region_Down,
    int region_Up)
{
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    int2 coord = (int2)(get_global_id(0), get_global_id(1));

    if (coord.y > region_Up && coord.y < (height - region_Down)){
        uint4 pixel = read_imageui(image, sampler, coord);

        // If pixel is white
        if (pixel.x == 255 && pixel.y == 255 && pixel.z == 255) {
            double theta;
            int rho;

            // Left line
            for (int i = 0; i < 55; i++) {
                theta = i * (M_PI / 180);  // theta in rads
                rho = coord.x * cos(theta) + coord.y * sin(theta);
                atomic_add(&HS1[(rho * 180 + i)], 1);
            }

            // Right line
            for (int i = 125; i < 180; i++) {
                theta = i * (M_PI / 180);
                rho = (width - coord.x) * -cos(theta) + coord.y * sin(theta);
                atomic_add(&HS2[(rho * 180 + i)], 1);
            }
        }
    }
}

/*__kernel void edgePixels_houghSpace(
    __read_only image2d_t src,
    __global int* HS1,
    int width,
    int maxAngle,
    __write_only image2d_t dst)
{
    const sampler_t samp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    // utilizar o sobel para um maior realce
    float kernel_diag1[3][3] = {{0,1,2}, {-1,0,1}, {-2,-1,0}};
    float kernel_diag2[3][3] = {{-2,-1,0}, {-1,0,1}, {0,1,2}};

    int2 coord = (int2)(get_global_id(0), get_global_id(1));

    float magX = 0.0, magY = 0.0;

    for(int a = 0; a < 3; a++)
    {
        for(int b = 0; b < 3; b++)
        {
            int2 offset = (int2)(a - 1, b - 1);
            int2 sample_coord = coord + offset;

            uint4 pixel = read_imageui(src, samp, sample_coord);

            // Convert to grayscale
            float gray = 0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z;

            magX += gray * kernel_diag1[a][b];
            magY += gray * kernel_diag2[a][b];
        }
    }

    float mag = sqrt(magX*magX + magY*magY)*1.5;
    if(mag<150){mag=0;}
    if(mag>255){mag=255;}
    //if(coord.y > 600) {mag=0;}
    if(coord.y < 300 || coord.y > 600) {mag=0;}
    uint4 output_pixel = (uint4)(mag, mag, mag, 255);

    //threshold
    if (mag > 150) {
        float A1[180];
        //float A2[21];

        for(int i = 0; i < maxAngle; i++) {
           A1[i] = M_PI / 180.0 * (i);
           //A2[i] = tan(M_PI / 180.0 * (10 - i));
        }

        for(int i = 0; i < maxAngle; i++) {
            float theta = A1[i];
            float rho = coord.x * cos(theta) + coord.y * sin(theta); //vertical

            atomic_add(&HS1[(int)(rho * maxAngle + i)], 1);
            //atomic_add(&HS1[(int)rho][i], 1); indexation of multi-array não funciona em opencl
        }
    }

    uint4 pixel = read_imageui(src, samp, coord);
    write_imageui(dst, coord, output_pixel);
}


__kernel void find_lines(__global int* HS1, __global int2* lines, int maxLines, int maxDistance, int maxAngle) {
    int id = get_global_id(0);

    if(id < maxLines) {
        int maxVotes = 0;
        int2 maxLine = (int2)(0, 0);

        int start = id * (maxDistance / maxLines);
        int end = start + (maxDistance / maxLines);

        for(int rho = start; rho < end; rho++) {
            for(int i = 0; i < maxAngle; i++) {
                int votes = HS1[rho * maxAngle + i];
                //int votes = HS1[rho][theta];

                if(votes > maxVotes) {
                    maxVotes = votes;
                    float theta = i;
                    maxLine = (int2)(rho, theta);
                }
            }
        }

        lines[id] = maxLine;
    }
}*/
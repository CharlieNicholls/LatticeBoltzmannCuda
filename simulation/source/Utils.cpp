
namespace Utils
{
    unsigned int compressInts(int* array, const int num, const int bitLength)
    {
        unsigned int output = 0;
        unsigned int max = 1;

        for(int bit = 0; bit < bitLength; ++bit)
        {
            max *= 2;
        }

        for(int i = 0; i < num - 1; ++i)
        {
            if(array[i] < max)
            {
                output = output | array[i];
                output = output << 5;
            }
        }

        if(array[num - 1] < max)
        {
            output = output | array[num - 1];
        }

        return output;
    }
}
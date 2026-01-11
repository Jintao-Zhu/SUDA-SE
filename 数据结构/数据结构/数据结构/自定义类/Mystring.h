#ifndef MYSTRING
#define MYSTRING

#include <iostream>

const int MaxSize = 1000;
class String
{
public:
    String() : data{0}, length{0} {}

    String(const char *s)
    {
        length = 0;
        while (s[length] != '\0')
        {
            length++;
        }
        data = new char[length];
        for (int i{0}; i < length; i++)
        {
            data[i] = s[i];
        }
    }

    String(const String &str) : length{str.length}
    {
        data = new char[length];
        for (int i{0}; i < length; i++)
        {
            data[i] = str.data[i];
        }
    }

    ~String()
    {
        delete[] data;
    }

    String &operator=(const String &other)
    {
        if (this != &other)
        {
            delete[] data;
            length = other.length;
            data = new char[length];
            for (int i{0}; i < length; i++)
            {
                data[i] = other.data[i];
            }
        }
        return *this;
    }

    char &operator[](int index)
    {
        return data[index];
    }

private:
    char *data;
    int length;
};

#endif
#include "cereal/cereal.hpp"
#include "cereal/types/unordered_map.hpp"
#include "cereal/types/memory.hpp"
#include "cereal/archives/binary.hpp"
#include "cereal/types/vector.hpp"
#include "cereal/types/map.hpp"

#include "custom_common.h"
#include <sys/stat.h>
#include <fcntl.h>
#include "hiaiengine/data_type_reg.h"


template<class Archive>
void serialize(Archive& ar, CustomFileBlob& data)
{
    ar(data.size);
    if (data.size > 0 && data.data.get() == nullptr) {
        data.data.reset(new char[data.size]);
    }
    ar(cereal::binary_data(data.data.get(), data.size * sizeof(char)));
}


template<class Archive>
void serialize(Archive& ar, CustomInfo& info)
{
    ar(info.name, info.type, info.outputSizeList, info.binFile, info.inputList,
       info.configFile,
       info.dataTypeList,
       info.precisionDeviation,
       info.statisticalDiscrepancy,
       info.expectFileList);
}

template<class Archive>
void serialize(Archive& ar, CustomOutput& info)
{
    ar(info.size, info.outputList, info.compareResultList);
}

HIAI_REGISTER_DATA_TYPE("CustomFileBlob", CustomFileBlob)
HIAI_REGISTER_DATA_TYPE("CustomInfo", CustomInfo)
HIAI_REGISTER_DATA_TYPE("CustomOutput", CustomOutput)

int32_t  WriteFile(const char* fileName, const char* buffer, uint32_t size)
{
    if (fileName == nullptr || buffer == nullptr) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "Invalid writeFile param!");
        return -1;
    }
    HIAI_ENGINE_LOG(HIAI_IDE_INFO, "fileName : %s", fileName);
    int fd = open(fileName, O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWRITE);
    if (fd < 0) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "Open file Failed when writeFile!");
        return -1;
    }
    uint32_t writeSize = write(fd, buffer, size);
    if (writeSize != size) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "Write file Failed!");
        close(fd);
        fd = -1;
        return -1;
    }
    close(fd);
    fd = -1;
    return 0;
}

char* ReadFile(const char *fileName, uint32_t *fileSize)
{
    if (fileName == nullptr || fileSize == nullptr) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "Invalid readFile param!");
        return nullptr;
    }
    std::ifstream file;
    file.open(fileName, std::ios::binary);
    if (!file.is_open()) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "Open file Failed when readFile!");
        return nullptr;
    }
    std::filebuf *pbuf = file.rdbuf();
    size_t size = pbuf->pubseekoff(0, std::ios::end, std::ios::in);
    pbuf->pubseekpos(0, std::ios::in);
    char *buffer = new (std::nothrow) char[size];
    if (buffer == nullptr) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "New buffer failed!");
        file.close();
        return nullptr;
    }

    pbuf->sgetn(buffer, size);
    *fileSize = size;

    file.close();
    return buffer;
}



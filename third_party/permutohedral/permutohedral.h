// This code is from https://andrew.adams.pub/thesis.pdf
#ifndef PERMUTOHEDRAL_H
#define PERMUTOHEDRAL_H

#include <math.h>
#include <vector>
#include <memory>
using std::vector;

// Hash table implementation for permutohedral lattice.
//
// The lattice points are stored sparsely using a hash table.
// The key for each point is its spatial location in the (d+1)-
// dimensional space.
class HashTablePermutohedral
{
  public:
    // Hash table constructor
    // kd : the dimensionality of the position vectors
    // vd : the dimensionality of the value vectors
    HashTablePermutohedral(int kd, int vd) : kd(kd), vd(vd)
    {
        filled = 0;
        entries.resize(1 << 15);
        keys.resize(kd * entries.size() / 2);
        values.resize(vd * entries.size() / 2, 0.0f);
    }
    // Returns the number of vectors stored.
    int size() { return filled; }
    // Returns a pointer to the keys array.
    vector<short> &getKeys() { return keys; }
    // Returns a pointer to the values array.
    vector<float> &getValues() { return values; }

    // Looks up the value vector associated with a given key. May or
    // may not create a new entry if that key doesn’t exist.
    float *lookup(const vector<short> &key, bool create = true)
    {
        // Double hash table size if necessary
        if (create && filled >= entries.size() / 2)
        {
            grow();
        }
        // Hash the key
        size_t h = hash(&key[0]) % entries.size();
        // Find the entry with the given key

        while (1)
        {
            Entry e = entries[h];
            // Check if the cell is empty
            if (e.keyIdx == -1)
            {
                if (!create)
                    return NULL;
                // Not found
                // Need to create an entry. Store the given key.
                for (int i = 0; i < kd; i++)
                {
                    keys[filled * kd + i] = key[i];
                }
                e.keyIdx = filled * kd;
                e.valueIdx = filled * vd;
                entries[h] = e;
                filled++;
                return &values[e.valueIdx];
            }
            // check if the cell has a matching key
            bool match = true;
            for (int i = 0; i < kd && match; i++)
            {
                match = keys[e.keyIdx + i] == key[i];
            }
            if (match)
            {
                return &values[e.valueIdx];
            }
            // increment the bucket with wraparound
            h++;
            if (h == entries.size())
            {
                h = 0;
            }
        }
    }

    // Hash function used in this implementation. A simple base conversion.
    size_t hash(const short *key)
    {
        size_t h = 0;
        for (int i = 0; i < kd; i++)
        {
            h += key[i];
            h *= 2531011;
        }
        return h;
    }

  private:
    // Grows the hash table when it runs out of space
    void grow()
    {
        // Grow the arrays
        values.resize(vd * entries.size(), 0.0f);
        keys.resize(kd * entries.size());
        vector<Entry> newEntries(entries.size() * 2);
        // Rehash all the entries
        for (size_t i = 0; i < entries.size(); i++)
        {
            if (entries[i].keyIdx == -1)
            {
                continue;
            }
            size_t h = hash(&keys[entries[i].keyIdx]) % newEntries.size();
            while (newEntries[h].keyIdx != -1)
            {
                h++;
                if (h == newEntries.size())
                {
                    h = 0;
                }
            }
            newEntries[h] = entries[i];
        }
        entries.swap(newEntries);
    }

    // Private struct for the hash table entries.
    struct Entry
    {
        Entry() : keyIdx(-1), valueIdx(-1) {}
        int keyIdx;
        int valueIdx;
    };
    vector<short> keys;
    vector<float> values;
    vector<Entry> entries;
    size_t filled;
    int kd, vd;
};

// The algorithm class that performs the filter
//
// PermutohedralLattice::filter(...) does all the work.
//
class PermutohedralLattice
{
  public:
    // Performs a Gauss transform
    // pos : position vectors
    // pd : position dimensions
    // val : value vectors
    // vd : value dimensions
    // n : number of items to filter
    // out : place to store the output
    static void filter(const float *pos, int pd,
                       const float *val, int vd,
                       int n, float *out)
    {
        // Create lattice
        PermutohedralLattice lattice(pd, vd, n);
        // Splat
        for (int i = 0; i < n; i++)
        {
            lattice.splat(pos + i * pd, val + i * vd);
        }
        // Blur
        lattice.blur();
        // Slice
        lattice.beginSlice();
        for (int i = 0; i < n; i++)
        {
            lattice.slice(out + i * vd);
        }
    }
    // Permutohedral lattice constructor
    // pd : dimensionality of position vectors
    // vd : dimensionality of value vectors
    // n : number of points in the input
    PermutohedralLattice(int pd, int vd, int n) : d(pd), vd(vd), n(n), hashTable(pd, vd)
    {
        // Allocate storage for various arrays
        elevated.resize(d + 1);
        scaleFactor.resize(d);
        greedy.resize(d + 1);
        rank.resize(d + 1);
        barycentric.resize(d + 2);
        canonical.resize((d + 1) * (d + 1));
        key.resize(d + 1);
        replay.resize(n * (d + 1));
        nReplay = 0;
        // compute the coordinates of the canonical simplex, in which
        // the difference between a contained point and the zero
        // remainder vertex is always in ascending order.
        for (int i = 0; i <= d; i++)
        {
            for (int j = 0; j <= d - i; j++)
            {
                canonical[i * (d + 1) + j] = i;
            }
            for (int j = d - i + 1; j <= d; j++)
            {
                canonical[i * (d + 1) + j] = i - (d + 1);
            }
        }
        // Compute part of the rotation matrix E that elevates a
        // position vector into the hyperplane
        for (int i = 0; i < d; i++)
        {
            // the diagonal entries for normalization
            scaleFactor[i] = 1.0f / (sqrtf((float)(i + 1) * (i + 2)));
            // We presume that the user would like to do a Gaussian
            // blur of standard deviation 1 in each dimension (or a
            // total variance of d, summed over dimensions.) Because
            // the total variance of the blur performed by this
            // algorithm is not d, we must scale the space to offset
            // this.
            // The total variance of the algorithm is:
            // [variance of splatting] +
            // [variance of blurring] +
            // [variance of splatting]
            // = d(d+1)(d+1)/12 + d(d+1)(d+1)/2 + d(d+1)(d+1)/12
            // = 2d(d+1)(d+1)/3.
            // So we need to scale the space by (d+1)sqrt(2/3).
            scaleFactor[i] *= (d + 1) * sqrtf(2.0 / 3);
        }
    }
    // Performs splatting with given position and value vectors
    void splat(const float *position, const float *value)
    {
        // First elevate position into the (d+1)-dimensional hyperplane
        elevated[d] = -d * position[d - 1] * scaleFactor[d - 1];
        for (int i = d - 1; i > 0; i--)
            elevated[i] = (elevated[i + 1] -
                           i * position[i - 1] * scaleFactor[i - 1] +
                           (i + 2) * position[i] * scaleFactor[i]);
        elevated[0] = elevated[1] + 2 * position[0] * scaleFactor[0];
        // Prepare to find the closest lattice points
        float scale = 1.0f / (d + 1);
        // Greedily search for the closest remainder-zero lattice point
        int sum = 0;
        for (int i = 0; i <= d; i++)
        {
            float v = elevated[i] * scale;
            float up = ceilf(v) * (d + 1);
            float down = floorf(v) * (d + 1);
            if (up - elevated[i] < elevated[i] - down)
            {
                greedy[i] = (short)up;
            }
            else
            {
                greedy[i] = (short)down;
            }
            sum += greedy[i];
        }
        sum /= d + 1;
        // Rank differential to find the permutation between this
        // simplex and the canonical one.
        for (int i = 0; i < d + 1; i++)
            rank[i] = 0;
        for (int i = 0; i < d; i++)
        {
            for (int j = i + 1; j <= d; j++)
            {
                if (elevated[i] - greedy[i] < elevated[j] - greedy[j])
                {
                    rank[i]++;
                }
                else
                {
                    rank[j]++;
                }
            }
        }
        if (sum > 0)
        {
            // Sum too large - the point is off the hyperplane. We
            // need to bring down the ones with the smallest
            // differential
            for (int i = 0; i <= d; i++)
            {
                if (rank[i] >= d + 1 - sum)
                {
                    greedy[i] -= d + 1;
                    rank[i] += sum - (d + 1);
                }
                else
                {
                    rank[i] += sum;
                }
            }
        }
        else if (sum < 0)
        {
            // Sum too small - the point is off the hyperplane. We
            // need to bring up the ones with largest differential
            for (int i = 0; i <= d; i++)
            {
                if (rank[i] < -sum)
                {
                    greedy[i] += d + 1;
                    rank[i] += (d + 1) + sum;
                }
                else
                {
                    rank[i] += sum;
                }
            }
        }
        // Compute barycentric coordinates
        for (int i = 0; i < d + 2; i++)
        {
            barycentric[i] = 0.0f;
        }
        for (int i = 0; i <= d; i++)
        {
            barycentric[d - rank[i]] += (elevated[i] - greedy[i]) * scale;
            barycentric[d + 1 - rank[i]] -= (elevated[i] - greedy[i]) * scale;
        }
        barycentric[0] += 1.0f + barycentric[d + 1];
        // Splat the value into each vertex of the simplex, with
        // barycentric weights
        for (int remainder = 0; remainder <= d; remainder++)
        {
            // Compute the location of the lattice point explicitly
            // (all but the last coordinate - it’s redundant because
            // they sum to zero)
            for (int i = 0; i < d; i++)
            {
                key[i] = greedy[i] + canonical[remainder * (d + 1) + rank[i]];
            }
            // Retrieve pointer to the value at this vertex
            float *val = hashTable.lookup(key, true);
            // Accumulate values with barycentric weight
            for (int i = 0; i < vd; i++)
            {
                val[i] += barycentric[remainder] * value[i];
            }
            // Record this interaction to use later when slicing
            replay[nReplay].offset = val - &hashTable.getValues()[0];
            replay[nReplay].weight = barycentric[remainder];
            nReplay++;
        }
    }
    // Prepare for slicing
    void beginSlice()
    {
        nReplay = 0;
    }
    // Performs slicing out of position vectors. The barycentric
    // weights and the simplex containing each position vector were
    // calculated and stored in the splatting step.
    void slice(float *col)
    {
        const vector<float> &vals = hashTable.getValues();
        for (int j = 0; j < vd; j++)
        {
            col[j] = 0;
        }
        for (int i = 0; i <= d; i++)
        {
            ReplayEntry r = replay[nReplay++];
            for (int j = 0; j < vd; j++)
            {
                col[j] += r.weight * vals[r.offset + j];
            }
        }
    }
    // Performs a Gaussian blur along each projected axis in the hyperplane.
    void blur()
    {
        // Prepare temporary arrays
        vector<short> neighbor1(d + 1), neighbor2(d + 1);
        vector<float> zero(vd, 0.0f);
        vector<float> newValue(vd * hashTable.size());
        vector<float> &oldValue = hashTable.getValues();
        // For each of d+1 axes,
        for (int j = 0; j <= d; j++)
        {
            // For each vertex in the lattice,
            for (int i = 0; i < hashTable.size(); i++)
            {
                // Blur point i in dimension j
                short *key = &(hashTable.getKeys()[i * d]);
                for (int k = 0; k < d; k++)
                {
                    neighbor1[k] = key[k] + 1;
                    neighbor2[k] = key[k] - 1;
                }
                neighbor1[j] = key[j] - d;
                neighbor2[j] = key[j] + d;
                float *oldVal = &oldValue[i * vd];
                float *newVal = &newValue[i * vd];
                float *v1 = hashTable.lookup(neighbor1, false);
                float *v2 = hashTable.lookup(neighbor2, false);
                if (!v1)
                    v1 = &zero[0];
                if (!v2)
                    v2 = &zero[0];
                // Mix values of the three vertices
                for (int k = 0; k < vd; k++)
                {
                    newVal[k] = (v1[k] + 2 * oldVal[k] + v2[k]);
                }
            }
            newValue.swap(oldValue);
        }
    }

  private:
    int d, vd, n;
    vector<float> elevated, scaleFactor, barycentric;
    vector<short> canonical, key, greedy;
    vector<char> rank;
    struct ReplayEntry
    {
        int offset;
        float weight;
    };
    vector<ReplayEntry> replay;
    int nReplay;
    HashTablePermutohedral hashTable;
};
#endif

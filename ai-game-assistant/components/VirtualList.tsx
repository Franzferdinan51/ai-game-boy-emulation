import React, { useRef, useEffect, useCallback, useMemo } from 'react';
import { motion } from 'framer-motion';
import { useInView } from 'react-intersection-observer';

interface VirtualListProps<T> {
  items: T[];
  renderItem: (item: T, index: number) => React.ReactNode;
  itemHeight: number | ((index: number) => number);
  overscan?: number;
  className?: string;
  onLoadMore?: () => void;
  hasMore?: boolean;
  loadingComponent?: React.ReactNode;
  emptyComponent?: React.ReactNode;
}

export function VirtualList<T>({
  items,
  renderItem,
  itemHeight,
  overscan = 5,
  className = '',
  onLoadMore,
  hasMore = false,
  loadingComponent,
  emptyComponent
}: VirtualListProps<T>) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [containerHeight, setContainerHeight] = useState(0);

  // Calculate item heights
  const getItemHeight = useCallback((index: number) => {
    return typeof itemHeight === 'function' ? itemHeight(index) : itemHeight;
  }, [itemHeight]);

  // Calculate total height of all items
  const totalHeight = useMemo(() => {
    return items.reduce((total, _, index) => total + getItemHeight(index), 0);
  }, [items, getItemHeight]);

  // Calculate visible range
  const visibleRange = useMemo(() => {
    if (containerHeight === 0) return { start: 0, end: 0 };

    let start = 0;
    let end = 0;
    let accumulatedHeight = 0;

    // Find start index
    for (let i = 0; i < items.length; i++) {
      const height = getItemHeight(i);
      if (accumulatedHeight + height > scrollTop - overscan * getItemHeight(0)) {
        start = i;
        break;
      }
      accumulatedHeight += height;
    }

    // Find end index
    accumulatedHeight = 0;
    for (let i = 0; i < items.length; i++) {
      const height = getItemHeight(i);
      accumulatedHeight += height;
      if (accumulatedHeight > scrollTop + containerHeight + overscan * getItemHeight(0)) {
        end = i + 1;
        break;
      }
    }

    return { start, end: end || items.length };
  }, [scrollTop, containerHeight, items, getItemHeight, overscan]);

  // Handle scroll events
  const handleScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
    const scrollTop = e.currentTarget.scrollTop;
    setScrollTop(scrollTop);

    // Check if we should load more
    if (onLoadMore && hasMore) {
      const scrollHeight = e.currentTarget.scrollHeight;
      const clientHeight = e.currentTarget.clientHeight;
      const scrollPercentage = (scrollTop + clientHeight) / scrollHeight;

      if (scrollPercentage > 0.8) {
        onLoadMore();
      }
    }
  }, [onLoadMore, hasMore]);

  // Update container height
  useEffect(() => {
    const updateHeight = () => {
      if (containerRef.current) {
        setContainerHeight(containerRef.current.clientHeight);
      }
    };

    updateHeight();
    window.addEventListener('resize', updateHeight);
    return () => window.removeEventListener('resize', updateHeight);
  }, []);

  // Calculate offset for each item
  const getItemOffset = useCallback((index: number) => {
    let offset = 0;
    for (let i = 0; i < index; i++) {
      offset += getItemHeight(i);
    }
    return offset;
  }, [getItemHeight]);

  const visibleItems = items.slice(visibleRange.start, visibleRange.end);

  return (
    <div
      ref={containerRef}
      className={`overflow-y-auto ${className}`}
      onScroll={handleScroll}
      style={{ height: '100%' }}
    >
      <div style={{ height: totalHeight, position: 'relative' }}>
        {visibleItems.map((item, index) => {
          const actualIndex = visibleRange.start + index;
          const offset = getItemOffset(actualIndex);
          const height = getItemHeight(actualIndex);

          return (
            <div
              key={actualIndex}
              style={{
                position: 'absolute',
                top: offset,
                left: 0,
                right: 0,
                height: height,
              }}
            >
              {renderItem(item, actualIndex)}
            </div>
          );
        })}
      </div>

      {items.length === 0 && emptyComponent && (
        <div className="flex items-center justify-center h-full">
          {emptyComponent}
        </div>
      )}

      {hasMore && loadingComponent && (
        <div className="flex items-center justify-center p-4">
          {loadingComponent}
        </div>
      )}
    </div>
  );
}

// Lazy loading component
interface LazyLoadProps {
  children: React.ReactNode;
  placeholder?: React.ReactNode;
  offset?: number;
  className?: string;
}

export const LazyLoad: React.FC<LazyLoadProps> = ({
  children,
  placeholder,
  offset = 100,
  className = ''
}) => {
  const { ref, inView } = useInView({
    triggerOnce: true,
    rootMargin: `${offset}px`,
  });

  return (
    <div ref={ref} className={className}>
      {inView ? children : placeholder}
    </div>
  );
};

// Infinite scroll component
interface InfiniteScrollProps {
  children: React.ReactNode;
  onLoadMore: () => void;
  hasMore: boolean;
  loading: boolean;
  threshold?: number;
  className?: string;
}

export const InfiniteScroll: React.FC<InfiniteScrollProps> = ({
  children,
  onLoadMore,
  hasMore,
  loading,
  threshold = 100,
  className = ''
}) => {
  const { ref, inView } = useInView({
    onChange: (inView) => {
      if (inView && hasMore && !loading) {
        onLoadMore();
      }
    },
    rootMargin: `${threshold}px`,
  });

  return (
    <div className={className}>
      {children}
      {hasMore && (
        <div ref={ref} className="flex justify-center p-4">
          {loading && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex items-center space-x-2 text-gray-400"
            >
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-cyan-500"></div>
              <span className="text-sm">Loading more...</span>
            </motion.div>
          )}
        </div>
      )}
    </div>
  );
};

// Image with lazy loading
interface LazyImageProps extends React.ImgHTMLAttributes<HTMLImageElement> {
  src: string;
  alt: string;
  placeholder?: React.ReactNode;
  blurDataURL?: string;
}

export const LazyImage: React.FC<LazyImageProps> = ({
  src,
  alt,
  placeholder,
  blurDataURL,
  className = '',
  ...props
}) => {
  const [isLoaded, setIsLoaded] = React.useState(false);
  const [hasError, setHasError] = React.useState(false);
  const { ref, inView } = useInView({
    triggerOnce: true,
    rootMargin: '50px',
  });

  const handleLoad = () => setIsLoaded(true);
  const handleError = () => setHasError(true);

  return (
    <div ref={ref} className={`relative overflow-hidden ${className}`}>
      {inView && (
        <>
          {blurDataURL && (
            <img
              src={blurDataURL}
              alt=""
              className={`absolute inset-0 w-full h-full object-cover transition-opacity duration-300 ${
                isLoaded ? 'opacity-0' : 'opacity-100'
              }`}
            />
          )}
          <img
            src={src}
            alt={alt}
            onLoad={handleLoad}
            onError={handleError}
            className={`w-full h-full object-cover transition-opacity duration-300 ${
              isLoaded ? 'opacity-100' : 'opacity-0'
            }`}
            {...props}
          />
          {!isLoaded && !hasError && placeholder && (
            <div className="absolute inset-0 flex items-center justify-center bg-gray-800">
              {placeholder}
            </div>
          )}
          {hasError && (
            <div className="absolute inset-0 flex items-center justify-center bg-gray-800 text-gray-400">
              <div className="text-center">
                <div className="text-2xl mb-2">🖼️</div>
                <p className="text-sm">Failed to load image</p>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};